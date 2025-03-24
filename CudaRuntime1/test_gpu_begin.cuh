#include <stdio.h>
#include <cuda_runtime.h>
#include "gpu_define.cuh"
#include "gpu_person.cuh"
#include "gpu_utils.cuh"
#include "gpu_aleat.cuh"
#include "gpu_begin.cuh"

int main() {
    // Set up a small test grid
    const int L = 20;
    const int gridSize = (L + 2) * (L + 2);

    // Print the host constants to verify
    // At the beginning of main, add:
    printf("Host constants: S=%d, E=%d, IP=%d, IA=%d, ISLight=%d, ISModerate=%d, ISSevere=%d\n",
        S, E, IP, IA, ISLight, ISModerate, ISSevere);

    // Set up GPU constants
    setupGPUConstants();

    // Initialize RNG
    setupRNG(gridSize);

    printf("Testing population initialization with %dx%d grid\n", L, L);

    // Allocate host and device memory
    GPUPerson* h_population = new GPUPerson[gridSize];
    GPUPerson* d_population;
    cudaMalloc(&d_population, gridSize * sizeof(GPUPerson));

    // Allocate counters
    int* h_stateCounts = new int[15]();
    int* h_newCounts = new int[15]();
    int* d_stateCounts, * d_newCounts;
    cudaMalloc(&d_stateCounts, 15 * sizeof(int));
    cudaMalloc(&d_newCounts, 15 * sizeof(int));

    // Initialize counters
    cudaMemset(d_stateCounts, 0, 15 * sizeof(int));
    cudaMemset(d_newCounts, 0, 15 * sizeof(int));
    h_stateCounts[S] = L * L;  // All start as susceptible
    cudaMemcpy(d_stateCounts, h_stateCounts, 11 * sizeof(int), cudaMemcpyHostToDevice);

    // Launch population initialization
    int blockSize = 256;
    int numBlocks = (gridSize + blockSize - 1) / blockSize;

    printf("Launching simplified initPopulation_kernel with %d blocks of %d threads\n",
        numBlocks, blockSize);

    // Use the simplified initialization kernel
    initPopulation_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in population init: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceSynchronize();

    // Copy population to host for analysis
    cudaMemcpy(h_population, d_population, gridSize * sizeof(GPUPerson), cudaMemcpyDeviceToHost);

    // Print age distribution
    printf("\nAges distribution:\n");
    int ageGroups[10] = { 0 }; // 0-9, 10-19, ..., 90+
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            int idx = to1D(i, j, L);
            int ageGroup = h_population[idx].AgeYears / 10;
            if (ageGroup > 9) ageGroup = 9;
            ageGroups[ageGroup]++;
        }
    }

    for (int i = 0; i < 10; i++) {
        printf("  Age %2d-%2d: %4d people (%.1f%%)\n",
            i * 10, i * 10 + 9, ageGroups[i],
            100.0 * ageGroups[i] / (L * L));
    }

    // Distribute initial infections
    const int Eini = 5;
    const int IPini = 3;
    const int IAini = 2;
    const int ISLightini = 1;
    const int ISModerateini = 1;
    const int ISSevereini = 1;

    printf("\nDistributing initial infections: E=%d, IP=%d, IA=%d, ISLight=%d, ISModerate=%d, ISSevere=%d\n",
        Eini, IPini, IAini, ISLightini, ISModerateini, ISSevereini);

    // Launch infection distribution kernel
    distributeInitialInfections_kernel << <1, 1 >> > (d_population, d_rngStates,
        d_stateCounts, d_newCounts, L,
        Eini, IPini, IAini, ISLightini,
        ISModerateini, ISSevereini);

    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in infection distribution: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_population, d_population, gridSize * sizeof(GPUPerson), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_stateCounts, d_stateCounts, 15 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_newCounts, d_newCounts, 15 * sizeof(int), cudaMemcpyDeviceToHost);

    // Count health states more carefully
    printf("\nHealth states after infection distribution:\n");
    int states[15] = { 0 };  // Indexes 0-14

    // Print each individual to help debug
    printf("\nIndividuals with special states:\n");
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            int idx = to1D(i, j, L);
            int health = h_population[idx].Health;

            // Count in our array
            if (health >= 0 && health < 15) {
                states[health]++;
            }

            // Print special cases for debugging
            if (health == ISLight || health == ISModerate || health == ISSevere) {
                printf("Found individual at (%d,%d) with health state %d\n", i, j, health);
            }
        }
    }
    // Print health states
    printf("  S: %d (expected: %d)\n", states[S], L * L - (Eini + IPini + IAini + ISLightini + ISModerateini + ISSevereini));
    printf("  E: %d (expected: %d)\n", states[E], Eini);
    printf("  IP: %d (expected: %d)\n", states[IP], IPini);
    printf("  IA: %d (expected: %d)\n", states[IA], IAini);
    printf("  ISLight: %d (expected: %d)\n", states[ISLight], ISLightini);
    printf("  ISModerate: %d (expected: %d)\n", states[ISModerate], ISModerateini);
    printf("  ISSevere: %d (expected: %d)\n", states[ISSevere], ISSevereini);

    // Compare with counter values
    printf("\nCounter values:\n");
    printf("  S: %d (new: %d)\n", h_stateCounts[S], h_newCounts[S]);
    printf("  E: %d (new: %d)\n", h_stateCounts[E], h_newCounts[E]);
    printf("  IP: %d (new: %d)\n", h_stateCounts[IP], h_newCounts[IP]);
    printf("  IA: %d (new: %d)\n", h_stateCounts[IA], h_newCounts[IA]);
    printf("  ISLight: %d (new: %d)\n", h_stateCounts[ISLight], h_newCounts[ISLight]);
    printf("  ISModerate: %d (new: %d)\n", h_stateCounts[ISModerate], h_newCounts[ISModerate]);
    printf("  ISSevere: %d (new: %d)\n", h_stateCounts[ISSevere], h_newCounts[ISSevere]);

    // Clean up and exit
    delete[] h_population;
    delete[] h_stateCounts;
    delete[] h_newCounts;
    cudaFree(d_population);
    cudaFree(d_stateCounts);
    cudaFree(d_newCounts);
    cleanupRNG();
    cleanupGPUConstants();

    printf("\nTest completed successfully\n");
    return 0;
}