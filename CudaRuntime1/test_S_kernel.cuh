#include <stdio.h>
#include <cuda_runtime.h>
#include "gpu_define.cuh"
#include "gpu_person.cuh"
#include "gpu_utils.cuh"
#include "gpu_aleat.cuh"
#include "gpu_neighbors.cuh"

#include "S_kernel.cuh"

void test_S_state() {
    const int L = 5;
    const int gridSize = (L + 2) * (L + 2);

    // Allocate and initialize test population
    GPUPerson* h_population = new GPUPerson[gridSize];
    GPUPerson* d_population;
    cudaMalloc(&d_population, gridSize * sizeof(GPUPerson));

    // Initialize all as Susceptible
    printf("\nInitializing population...\n");
    for (int i = 0; i < gridSize; i++) {
        h_population[i].Health = S;
        h_population[i].Swap = S;
        h_population[i].TimeOnState = 0;
        h_population[i].StateTime = 0;
        h_population[i].Days = 0;
        h_population[i].AgeDeathDays = 365 * 80;  // 80 years
        h_population[i].Checked = 0;
        h_population[i].Exponent = 0;
    }

    // Add more detailed initial state print
    printf("\nInitial State Details:\n");
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            int idx = to1D(i, j, L);
            if (h_population[idx].Health != S) {
                printf("Infectious at (%d,%d):\n", i, j);
                printf("- Health state: %d\n", h_population[idx].Health);
                printf("- TimeOnState: %d\n", h_population[idx].TimeOnState);
            }
        }
    }

    // Add Beta and contact parameters print
    double verify_beta;
    cudaMemcpyFromSymbol(&verify_beta, d_Beta, sizeof(double));
    printf("\nInfection Parameters:\n");
    printf("Beta: %f\n", verify_beta);
    printf("Max Random Contacts: %f\n", MaxRandomContacts);
    printf("Min Random Contacts: %f\n", MinRandomContacts);

    // Set up test cases
    printf("\nSetting up test cases:\n");

    // Case 1: Susceptible with infectious neighbors
    printf("\nCase 1 - Susceptible with infectious neighbors (2,2):\n");
    h_population[to1D(2, 2, L)].Health = S;
    // Add infectious neighbors
    h_population[to1D(2, 3, L)].Health = IP;
    h_population[to1D(3, 2, L)].Health = IA;

    // Case 2: Susceptible about to die
    printf("\nCase 2 - Natural Death Case (3,3):\n");
    h_population[to1D(3, 3, L)].Health = S;
    h_population[to1D(3, 3, L)].Days = 365 * 85;  // 85 years

    // Case 3: Isolated susceptible
    printf("\nCase 3 - Isolated Susceptible (4,4):\n");
    h_population[to1D(4, 4, L)].Health = S;

    // Print initial state
    printf("\nInitial Grid State:\n");
    printf("S: Susceptible, I: Infectious\n");
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            char state = 'S';
            if (h_population[to1D(i, j, L)].Health == IP ||
                h_population[to1D(i, j, L)].Health == IA) state = 'I';
            printf("%c ", state);
        }
        printf("\n");
    }

    // Copy to device
    cudaMemcpy(d_population, h_population, gridSize * sizeof(GPUPerson),
        cudaMemcpyHostToDevice);

    // Setup RNG
    setupRNG(gridSize);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (gridSize + blockSize - 1) / blockSize;
    printf("\nLaunching S_kernel with %d blocks, %d threads per block\n",
        numBlocks, blockSize);

    printf("\nVerifying Neighbor Detection:\n");
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            int idx = to1D(i, j, L);
            if (h_population[idx].Health == S) {
                int local_infectious = 0;
                // Count local infectious neighbors
                for (int di = -1; di <= 1; di++) {
                    for (int dj = -1; dj <= 1; dj++) {
                        if (di == 0 && dj == 0) continue;
                        int ni = i + di;
                        int nj = j + dj;
                        if (ni > 0 && ni <= L && nj > 0 && nj <= L) {
                            int nidx = to1D(ni, nj, L);
                            if (isInfectious(h_population[nidx].Health)) {
                                local_infectious++;
                            }
                        }
                    }
                }
                if (local_infectious > 0) {
                    printf("Cell (%d,%d) has %d infectious neighbors\n",
                        i, j, local_infectious);
                }
            }
        }
    }

    S_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);

    // Get results back
    cudaMemcpy(h_population, d_population, gridSize * sizeof(GPUPerson),
        cudaMemcpyDeviceToHost);

    // Print final state
    printf("\nFinal Grid State:\n");
    printf("S: Susceptible, I: Infectious, E: Exposed, D: Dead\n");
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            char state = 'S';
            int idx = to1D(i, j, L);
            if (h_population[idx].Swap == E) state = 'E';
            else if (h_population[idx].Health == IP ||
                h_population[idx].Health == IA) state = 'I';
            else if (h_population[idx].Swap == Dead) state = 'D';
            printf("%c ", state);
        }
        printf("\n");
    }

    // Print detailed transitions
    printf("\nDetailed Final State for Test Cases:\n");

    printf("\nCase 1 (2,2) Final State:\n");
    printf("- Health: %d\n", h_population[to1D(2, 2, L)].Health);
    printf("- Swap: %d\n", h_population[to1D(2, 2, L)].Swap);
    printf("- Checked: %d\n", h_population[to1D(2, 2, L)].Checked);
    printf("- Exponent: %d\n", h_population[to1D(2, 2, L)].Exponent);
    printf("- TimeOnState: %d\n", h_population[to1D(2, 2, L)].TimeOnState);
    printf("- StateTime: %d\n", h_population[to1D(2, 2, L)].StateTime);

    printf("\nCase 2 (3,3) Final State:\n");
    printf("- Health: %d\n", h_population[to1D(3, 3, L)].Health);
    printf("- Swap: %d\n", h_population[to1D(3, 3, L)].Swap);
    printf("- Days: %d\n", h_population[to1D(3, 3, L)].Days);
    printf("- AgeDeathDays: %d\n", h_population[to1D(3, 3, L)].AgeDeathDays);

    printf("\nCase 3 (4,4) Final State:\n");
    printf("- Health: %d\n", h_population[to1D(4, 4, L)].Health);
    printf("- Swap: %d\n", h_population[to1D(4, 4, L)].Swap);
    printf("- Exponent: %d\n", h_population[to1D(4, 4, L)].Exponent);

    printf("\nTransition Statistics:\n");
    int total_exposed = 0;
    int total_dead = 0;
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            int idx = to1D(i, j, L);
            if (h_population[idx].Swap == E) total_exposed++;
            if (h_population[idx].Swap == Dead) total_dead++;
        }
    }
    printf("Total new exposed: %d\n", total_exposed);
    printf("Total new dead: %d\n", total_dead);

    // Cleanup
    delete[] h_population;
    cudaFree(d_population);
    cleanupRNG();
}

int main() {
    int city = ROC;
    setupCityParameters(city);
    setupGPUConstants();

    test_S_state();

    cleanupGPUConstants();
    return 0;
}