#include <stdio.h>
#include <cuda_runtime.h>
#include "gpu_define.cuh"
#include "gpu_person.cuh"
#include "gpu_utils.cuh"
#include "gpu_aleat.cuh"
#include "H_kernel.cuh"

int main() {
    // Set up a small test grid
    const int L = 10;
    const int gridSize = (L + 2) * (L + 2);

    // Set up GPU constants
    setupGPUConstants();

    // Initialize RNG
    setupRNG(gridSize);

    printf("Testing H_kernel with %dx%d grid\n", L, L);

    // Allocate host and device memory
    GPUPerson* h_population = new GPUPerson[gridSize];
    GPUPerson* d_population;
    cudaMalloc(&d_population, gridSize * sizeof(GPUPerson));

    // Set initial available beds
    int initialAvailableBeds = 5;
    int initialAvailableICUBeds = 2;
    cudaMemcpyToSymbol(AvailableBeds, &initialAvailableBeds, sizeof(int));
    cudaMemcpyToSymbol(AvailableBedsICU, &initialAvailableICUBeds, sizeof(int));

    // Initialize population
    for (int i = 0; i < gridSize; i++) {
        h_population[i].Health = S;
        h_population[i].Swap = S;
        h_population[i].TimeOnState = 0;
        h_population[i].StateTime = 0;
        h_population[i].Days = 0;
        h_population[i].AgeYears = 40;
        h_population[i].AgeDeathDays = 80 * 365;
    }

    // Set up test cases
    printf("\nSetting up test cases...\n");

    // Case 1: Young patient about to recover (high recovery chance)
    int idx = to1D(2, 2, L);
    h_population[idx].Health = H;
    h_population[idx].Swap = H;
    h_population[idx].AgeYears = 25;
    h_population[idx].StateTime = 10;
    h_population[idx].TimeOnState = 10; // Ready to transition
    printf("Case 1: Young patient (25yo) ready to transition at (2,2)\n");

    // Case 2: Elderly patient about to need ICU (low recovery chance)
    idx = to1D(3, 3, L);
    h_population[idx].Health = H;
    h_population[idx].Swap = H;
    h_population[idx].AgeYears = 75;
    h_population[idx].StateTime = 10;
    h_population[idx].TimeOnState = 10; // Ready to transition
    printf("Case 2: Elderly patient (75yo) ready to transition at (3,3)\n");

    // Case 3: Another elderly patient (ICU beds should run out)
    idx = to1D(4, 4, L);
    h_population[idx].Health = H;
    h_population[idx].Swap = H;
    h_population[idx].AgeYears = 80;
    h_population[idx].StateTime = 10;
    h_population[idx].TimeOnState = 10; // Ready to transition
    printf("Case 3: Elderly patient (80yo) ready to transition at (4,4)\n");

    // Case 4: Patient in middle of hospital stay
    idx = to1D(5, 5, L);
    h_population[idx].Health = H;
    h_population[idx].Swap = H;
    h_population[idx].AgeYears = 60;
    h_population[idx].StateTime = 20;
    h_population[idx].TimeOnState = 5; // Not ready to transition
    printf("Case 4: Patient (60yo) in middle of stay at (5,5)\n");

    // Case 5: Natural death case
    idx = to1D(6, 6, L);
    h_population[idx].Health = H;
    h_population[idx].Swap = H;
    h_population[idx].AgeYears = 70;
    h_population[idx].Days = 85 * 365; // Past death age
    h_population[idx].AgeDeathDays = 80 * 365;
    printf("Case 5: Patient (70yo) reaching natural death age at (6,6)\n");

    // Copy to device
    cudaMemcpy(d_population, h_population, gridSize * sizeof(GPUPerson), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (gridSize + blockSize - 1) / blockSize;
    printf("\nLaunching H_kernel with %d blocks of %d threads\n", numBlocks, blockSize);

    H_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(h_population, d_population, gridSize * sizeof(GPUPerson), cudaMemcpyDeviceToHost);

    // Get final bed counts
    int finalAvailableBeds, finalAvailableICUBeds;
    cudaMemcpyFromSymbol(&finalAvailableBeds, AvailableBeds, sizeof(int));
    cudaMemcpyFromSymbol(&finalAvailableICUBeds, AvailableBedsICU, sizeof(int));

    // Print results
    printf("\nTest Results:\n");
    printf("Initial Beds: Hospital=%d, ICU=%d\n", initialAvailableBeds, initialAvailableICUBeds);
    printf("Final Beds: Hospital=%d, ICU=%d\n", finalAvailableBeds, finalAvailableICUBeds);

    printf("\nPatient Outcomes:\n");

    // Case 1
    idx = to1D(2, 2, L);
    printf("Case 1 (2,2): %s (Health=%d, Swap=%d)\n",
        h_population[idx].Swap == Recovered ? "Recovered" :
        h_population[idx].Swap == ICU ? "Transferred to ICU" :
        h_population[idx].Swap == DeadCovid ? "Died (COVID)" :
        h_population[idx].Swap == Dead ? "Died (Natural)" : "Unknown",
        h_population[idx].Health, h_population[idx].Swap);

    // Case 2
    idx = to1D(3, 3, L);
    printf("Case 2 (3,3): %s (Health=%d, Swap=%d)\n",
        h_population[idx].Swap == Recovered ? "Recovered" :
        h_population[idx].Swap == ICU ? "Transferred to ICU" :
        h_population[idx].Swap == DeadCovid ? "Died (COVID)" :
        h_population[idx].Swap == Dead ? "Died (Natural)" : "Unknown",
        h_population[idx].Health, h_population[idx].Swap);

    // Case 3
    idx = to1D(4, 4, L);
    printf("Case 3 (4,4): %s (Health=%d, Swap=%d)\n",
        h_population[idx].Swap == Recovered ? "Recovered" :
        h_population[idx].Swap == ICU ? "Transferred to ICU" :
        h_population[idx].Swap == DeadCovid ? "Died (COVID)" :
        h_population[idx].Swap == Dead ? "Died (Natural)" : "Unknown",
        h_population[idx].Health, h_population[idx].Swap);

    // Case 4
    idx = to1D(5, 5, L);
    printf("Case 4 (5,5): %s (Health=%d, Swap=%d, TimeOnState=%d)\n",
        h_population[idx].Swap == H ? "Still in Hospital" : "Unexpected Transition",
        h_population[idx].Health, h_population[idx].Swap, h_population[idx].TimeOnState);

    // Case 5
    idx = to1D(6, 6, L);
    printf("Case 5 (6,6): %s (Health=%d, Swap=%d)\n",
        h_population[idx].Swap == Recovered ? "Recovered" :
        h_population[idx].Swap == ICU ? "Transferred to ICU" :
        h_population[idx].Swap == DeadCovid ? "Died (COVID)" :
        h_population[idx].Swap == Dead ? "Died (Natural)" : "Unknown",
        h_population[idx].Health, h_population[idx].Swap);

    // Clean up
    delete[] h_population;
    cudaFree(d_population);
    cleanupRNG();
    cleanupGPUConstants();

    printf("\nTest completed successfully\n");
    return 0;
}