// covid_simulation_minimal.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "gpu_define.cuh"
#include "gpu_person.cuh"
#include "gpu_utils.cuh"
#include "gpu_aleat.cuh"
#include "gpu_begin.cuh"
#include "update_kernel.cuh"

// Test with minimal functionality first
int main(int argc, char* argv[]) {
    printf("Starting minimal COVID-19 CUDA simulation test\n");

    // Initialize city and GPU constants
    int city = ROC;  // Rocinha
    setupCityParameters(city);
    setupGPUConstants();

    // Add simulation number parameter
    int simulationNumber = 1;

    // Basic parameters
    const int L = 632;  // Grid size
    const int gridSize = (L + 2) * (L + 2);
    const int N = L * L;

    printf("Grid size: %d x %d = %d cells\n", L, L, N);
    printf("Total array size (with boundaries): %d\n", gridSize);

    // Test 1: Allocate device memory
    printf("\n1. Testing device memory allocation...\n");
    GPUPerson* d_population;
    cudaError_t err = cudaMalloc(&d_population, gridSize * sizeof(GPUPerson));
    if (err != cudaSuccess) {
        printf("ERROR: Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        return -1;
    }
    printf("   SUCCESS: Allocated %zu MB for population\n",
        (gridSize * sizeof(GPUPerson)) / (1024 * 1024));

    // Test 2: Initialize RNG
    printf("\n2. Testing RNG initialization...\n");
    unsigned int* d_rngStates;
    err = cudaMalloc(&d_rngStates, gridSize * sizeof(unsigned int));
    if (err != cudaSuccess) {
        printf("ERROR: Failed to allocate RNG states: %s\n", cudaGetErrorString(err));
        cudaFree(d_population);
        return -1;
    }

    // Initialize RNG states with unique seed
    unsigned int seed = 893221891 * simulationNumber;
    int blockSize = 256;
    int numBlocks = (gridSize + blockSize - 1) / blockSize;

    initRNG << <numBlocks, blockSize >> > (d_rngStates, seed, gridSize);
    cudaDeviceSynchronize();

    // Verify RNG initialization
    unsigned int h_rngCheck[3];
    cudaMemcpy(h_rngCheck, d_rngStates, 3 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("   SUCCESS: RNG initialized (first 3 states: %u, %u, %u)\n",
        h_rngCheck[0], h_rngCheck[1], h_rngCheck[2]);

    // Test 3: Run population initialization kernel
    printf("\n3. Testing population initialization kernel...\n");


    initPopulation_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: Population init kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_population);
        cudaFree(d_rngStates);
        return -1;
    }
    cudaDeviceSynchronize();
    printf("   SUCCESS: Population initialized\n");

    // Test 4: Initialize counters
    printf("\n4. Testing counter initialization...\n");
    int* d_stateCounts, * d_newCounts;
    err = cudaMalloc(&d_stateCounts, 15 * sizeof(int));
    if (err == cudaSuccess) {
        err = cudaMalloc(&d_newCounts, 15 * sizeof(int));
    }
    if (err != cudaSuccess) {
        printf("ERROR: Failed to allocate counters: %s\n", cudaGetErrorString(err));
        cudaFree(d_population);
        cudaFree(d_rngStates);
        return -1;
    }

    initCounters_kernel << <1, 32 >> > (d_stateCounts, d_newCounts, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: Counter init kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_population);
        cudaFree(d_rngStates);
        cudaFree(d_stateCounts);
        cudaFree(d_newCounts);
        return -1;
    }
    cudaDeviceSynchronize();
    printf("   SUCCESS: Counters initialized\n");

    // Test 5: Distribute initial infections
    printf("\n5. Testing initial infection distribution...\n");
    distributeInitialInfections_kernel << <1, 1 >> > (
        d_population, d_rngStates, d_stateCounts, d_newCounts, L,
        0,  // Eini
        5,  // IPini
        0,  // IAini
        0,  // ISLightini
        0,  // ISModerateini
        0   // ISSevereini
        );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: Infection distribution kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_population);
        cudaFree(d_rngStates);
        cudaFree(d_stateCounts);
        cudaFree(d_newCounts);
        return -1;
    }
    cudaDeviceSynchronize();
    printf("   SUCCESS: Initial infections distributed\n");

    // Test 6: Get statistics
    printf("\n6. Testing statistics retrieval...\n");
    int h_totals[15] = { 0 };
    int h_new_cases[15] = { 0 };

    getCountersFromDevice(h_totals, h_new_cases);

    printf("   Initial population state:\n");
    printf("   S: %d\n", h_totals[S]);
    printf("   E: %d\n", h_totals[E]);
    printf("   IP: %d\n", h_totals[IP]);
    printf("   IA: %d\n", h_totals[IA]);
    printf("   Total: %d (should be %d)\n",
        h_totals[S] + h_totals[E] + h_totals[IP] + h_totals[IA], N);

    // Test 7: Run update kernel
    printf("\n7. Testing update kernel...\n");

    // Reset counters first
    resetCounters_kernel << <1, 1 >> > ();
    resetNewCounters_kernel << <1, 1 >> > ();
    cudaDeviceSynchronize();

    update_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L, 0, d_ProbNaturalDeath);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: Update kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_population);
        cudaFree(d_rngStates);
        cudaFree(d_stateCounts);
        cudaFree(d_newCounts);
        return -1;
    }
    cudaDeviceSynchronize();
    printf("   SUCCESS: Update kernel executed\n");

    // Get updated statistics
    getCountersFromDevice(h_totals, h_new_cases);
    printf("   After update:\n");
    printf("   S: %d (new: %d)\n", h_totals[S], h_new_cases[S]);
    printf("   IP: %d\n", h_totals[IP]);

    // Cleanup
    printf("\n8. Cleaning up...\n");
    cudaFree(d_population);
    cudaFree(d_rngStates);
    cudaFree(d_stateCounts);
    cudaFree(d_newCounts);
    cleanupGPUConstants();

    printf("\nMinimal test completed successfully!\n");

    return 0;
}