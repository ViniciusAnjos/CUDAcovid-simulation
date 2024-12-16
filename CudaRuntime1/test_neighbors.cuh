#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>

#include "gpu_define.cuh"
#include "gpu_person.cuh"
#include "gpu_utils.cuh"
#include "gpu_aleat.cuh"
#include "gpu_neighbors.cuh"

__global__ void test_neighbors_kernel(GPUPerson* population, ContactResult* results, unsigned int* rngStates, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * L) return;

    int i, j;
    to2D(idx, L, i, j);

    if (i > 0 && i <= L && j > 0 && j <= L) {
        unsigned int* myRNG = &rngStates[idx];
        results[idx] = checkAllContacts(i, j, L, population, myRNG);
    }
}

void test_neighbors_comprehensive() {
    const int L = 7;  // Larger grid for better testing
    const int gridSize = (L + 2) * (L + 2);

    // Allocate and initialize test population
    GPUPerson* h_population = new GPUPerson[gridSize];
    GPUPerson* d_population;
    cudaMalloc(&d_population, gridSize * sizeof(GPUPerson));

    printf("\n=== Testing Different Infectious States with Local and Random Contacts ===\n");

    // Initialize all as Susceptible
    for (int i = 0; i < gridSize; i++) {
        h_population[i].Health = S;
    }

    // Set up different infectious states in a pattern
    h_population[to1D(2, 2, L)].Health = IP;          // Pre-symptomatic
    h_population[to1D(2, 3, L)].Health = IA;          // Asymptomatic
    h_population[to1D(2, 4, L)].Health = ISLight;     // Light symptoms
    h_population[to1D(2, 5, L)].Health = ISModerate;  // Moderate symptoms
    h_population[to1D(3, 2, L)].Health = ISSevere;    // Severe symptoms
    h_population[to1D(3, 3, L)].Health = H;           // Hospitalized
    h_population[to1D(3, 4, L)].Health = ICU;         // ICU

    // Test with both density settings
    int densities[] = { LOW, HIGH };
    const char* density_names[] = { "LOW", "HIGH" };

    // Test with different random contact settings
    double test_max_contacts[] = { 2.5, 5.0, 119.5 };  // Testing different MaxRandomContacts
    double min_contacts = 1.5;  // MinRandomContacts remains constant

    for (int d = 0; d < 2; d++) {
        int density = densities[d];
        cudaMemcpyToSymbol(d_Density, &density, sizeof(int));

        for (int c = 0; c < 3; c++) {
            double max_contacts = test_max_contacts[c];
            cudaMemcpyToSymbol(d_MaxRandomContacts, &max_contacts, sizeof(double));
            cudaMemcpyToSymbol(d_MinRandomContacts, &min_contacts, sizeof(double));

            printf("\n=== Testing with %s Density and MaxRandomContacts = %.1f ===\n",
                density_names[d], max_contacts);

            // Copy population to device
            cudaMemcpy(d_population, h_population, gridSize * sizeof(GPUPerson), cudaMemcpyHostToDevice);

            // Allocate result array
            ContactResult* h_results = new ContactResult[gridSize];
            ContactResult* d_results;
            cudaMalloc(&d_results, gridSize * sizeof(ContactResult));

            // Setup RNG
            setupRNG(gridSize);

            // Launch test kernel
            int blockSize = 256;
            int numBlocks = (gridSize + blockSize - 1) / blockSize;
            test_neighbors_kernel << <numBlocks, blockSize >> > (d_population, d_results, d_rngStates, L);

            // Get results back
            cudaMemcpy(h_results, d_results, gridSize * sizeof(ContactResult), cudaMemcpyDeviceToHost);

            // Print grid layout
            printf("\nGrid layout (I=Various Infectious, S=Susceptible):\n");
            for (int i = 1; i <= L; i++) {
                for (int j = 1; j <= L; j++) {
                    int idx = to1D(i, j, L);
                    printf("%c ", h_population[idx].Health == S ? 'S' : 'I');
                }
                printf("\n");
            }

            // Print detailed state information
            printf("\nDetailed State Information:\n");
            for (int i = 1; i <= L; i++) {
                for (int j = 1; j <= L; j++) {
                    int idx = to1D(i, j, L);
                    if (h_population[idx].Health != S) {
                        const char* state;
                        switch (h_population[idx].Health) {
                        case IP: state = "IP"; break;
                        case IA: state = "IA"; break;
                        case ISLight: state = "ISLight"; break;
                        case ISModerate: state = "ISModerate"; break;
                        case ISSevere: state = "ISSevere"; break;
                        case H: state = "H"; break;
                        case ICU: state = "ICU"; break;
                        default: state = "Unknown"; break;
                        }
                        printf("Position (%d,%d): %s\n", i, j, state);
                    }
                }
            }

            // Print contact results
            printf("\nContact Results:\n");
            for (int i = 1; i <= L; i++) {
                for (int j = 1; j <= L; j++) {
                    int idx = to1D(i, j, L);
                    printf("Position (%d,%d): %d infectious contacts\n",
                        i, j, h_results[idx].infectiousContacts);
                }
            }

            // Cleanup per iteration
            delete[] h_results;
            cudaFree(d_results);
        }
    }

    // Final cleanup
    delete[] h_population;
    cudaFree(d_population);
    cleanupRNG();
}

int main() {
    // Initialize GPU constants
    setupGPUConstants();

    // Run comprehensive tests
    test_neighbors_comprehensive();

    // Cleanup GPU constants
    cleanupGPUConstants();

    printf("\nTest completed successfully!\n");

    return 0;
}