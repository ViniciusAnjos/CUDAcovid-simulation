#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>

#include "gpu_define.cuh"
#include "gpu_person.cuh"
#include "gpu_utils.cuh"
#include "gpu_aleat.cuh"
#include "gpu_neighbors.cuh"



// Modified kernel to test local contacts with proper RNG state handling
__global__ void testLocalContacts(GPUPerson* population, unsigned int* rngStates, int L) {
    // Each thread needs its own index and RNG state
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * L) return;  // Make sure we don't exceed grid size

    // Get thread's position in 2D grid
    int i = idx / L + 1;
    int j = idx % L + 1;

    // Only the first thread prints to avoid cluttered output
    if (idx == 0) {
        printf("\nTesting Local Contact Detection:\n");
        printf("--------------------------------\n");
        printf("Testing position (%d,%d):\n", i, j);

        // Print local neighborhood
        printf("Local neighborhood:\n");
        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                int ni = wrapIndex(i + di, L);
                int nj = wrapIndex(j + dj, L);
                int nIdx = to1D(ni, nj, L);
                printf("%d ", population[nIdx].Health);
            }
            printf("\n");
        }

        // Test contact checking
        ContactResult result = checkLocalContacts(i, j, L, population);
        printf("\nLocal contact results:\n");
        printf("Infectious contacts found: %d\n", result.infectiousContacts);
        printf("Any contact made: %s\n", result.anyContact ? "Yes" : "No");
    }
}

// Modified kernel to test random contacts with proper RNG state handling
__global__ void testRandomContacts(GPUPerson* population, unsigned int* rngStates, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * L) return;

    // Get this thread's RNG state
    unsigned int localState = rngStates[idx];

    // Only first thread prints results
    if (idx == 0) {
        printf("\nTesting Random Contact Generation:\n");
        printf("---------------------------------\n");

        int i = L / 2;
        int j = L / 2;

        // Run multiple random contact tests
        for (int test = 0; test < 5; test++) {
            ContactResult result = checkRandomContacts(i, j, L, population, &localState);
            printf("\nRandom contact test #%d:\n", test + 1);
            printf("Number of infectious contacts: %d\n", result.infectiousContacts);
            printf("Contact made: %s\n", result.anyContact ? "Yes" : "No");
        }
    }

    // Save back the RNG state
    rngStates[idx] = localState;
}

int main() {
    printf("Starting Contact System Test with Multiple RNG States\n");

    // Use a small grid for testing
    const int L = 10;
    const int totalSize = (L + 2) * (L + 2);
    const int activeSize = L * L;  // Number of active grid cells (excluding boundaries)
    cudaError_t err;

    // Allocate and initialize population
    GPUPerson* d_population;
    err = cudaMalloc(&d_population, totalSize * sizeof(GPUPerson));
    checkCudaError(err, "Failed to allocate population memory");

    // Set up test grid
    GPUPerson* h_population = new GPUPerson[totalSize];
    for (int i = 0; i < totalSize; i++) {
        h_population[i].Health = S;  // Initialize all as susceptible
        h_population[i].Isolation = IsolationNo;
    }

    // Add some infectious cases for testing
    int center_i = L / 2;
    int center_j = L / 2;
    h_population[to1D(center_i - 1, center_j, L)].Health = IP;
    h_population[to1D(center_i + 1, center_j, L)].Health = IA;
    h_population[to1D(center_i, center_j - 1, L)].Health = ISLight;
    h_population[to1D(center_i, center_j + 1, L)].Health = ISModerate;

    err = cudaMemcpy(d_population, h_population, totalSize * sizeof(GPUPerson),
        cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy population to device");

    // Initialize RNG states - one per active grid cell
    unsigned int* d_rngStates;
    err = cudaMalloc(&d_rngStates, activeSize * sizeof(unsigned int));
    checkCudaError(err, "Failed to allocate RNG states");

    // Set up RNG states
    int blockSize = 256;
    int numBlocks = (activeSize + blockSize - 1) / blockSize;
    initRNG << <numBlocks, blockSize >> > (d_rngStates, 12345, activeSize);
    err = cudaDeviceSynchronize();
    checkCudaError(err, "RNG initialization failed");

    // Initialize GPU constants
    setupGPUConstants();

    printf("\nTesting with grid size %dx%d\n", L, L);

    // Run test kernels
    testLocalContacts << <numBlocks, blockSize >> > (d_population, d_rngStates, L);
    err = cudaDeviceSynchronize();
    checkCudaError(err, "Local contacts test failed");

    testRandomContacts << <numBlocks, blockSize >> > (d_population, d_rngStates, L);
    err = cudaDeviceSynchronize();
    checkCudaError(err, "Random contacts test failed");

    // Clean up
    cudaFree(d_population);
    cudaFree(d_rngStates);
    delete[] h_population;

    printf("\nTest completed successfully!\n");
    return 0;
}