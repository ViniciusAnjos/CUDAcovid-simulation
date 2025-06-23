#include <stdio.h>
#include <cuda_runtime.h>
#include "gpu_define.cuh"
#include "gpu_Person.cuh"
#include "gpu_aleat.cuh"
#include "gpu_utils.cuh"
#include "H_kernel.cuh"

void test_H_kernel() {
    const int L = 5;
    const int gridSize = (L + 2) * (L + 2);

    // Allocate population
    GPUPerson* h_population = new GPUPerson[gridSize];
    GPUPerson* d_population;
    cudaMalloc(&d_population, gridSize * sizeof(GPUPerson));

    // Initialize all as Susceptible
    for (int i = 0; i < gridSize; i++) {
        h_population[i].Health = S;
        h_population[i].Swap = S;
        h_population[i].TimeOnState = 0;
        h_population[i].StateTime = 0;
        h_population[i].Days = 0;
        h_population[i].AgeYears = 40;
        h_population[i].AgeDeathDays = 365 * 80;
    }

    // Create and initialize device memory for bed counts
    int h_hospitalBeds = 5;
    int h_icuBeds = 2;

    int* d_hospitalBeds, * d_icuBeds;
    cudaMalloc(&d_hospitalBeds, sizeof(int));
    cudaMalloc(&d_icuBeds, sizeof(int));

    // Initialize with cudaMemcpy
    cudaMemcpy(d_hospitalBeds, &h_hospitalBeds, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_icuBeds, &h_icuBeds, sizeof(int), cudaMemcpyHostToDevice);

    // Verify values were set correctly with a readback
    int verify_hosp, verify_icu;
    cudaMemcpy(&verify_hosp, d_hospitalBeds, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&verify_icu, d_icuBeds, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Initial beds (verified): Hospital=%d, ICU=%d\n",
        verify_hosp, verify_icu);

    // Setup test cases - make sure Swap matches Health
    h_population[to1D(2, 2, L)].Health = H;
    h_population[to1D(2, 2, L)].Swap = H;
    h_population[to1D(2, 2, L)].StateTime = 5;
    h_population[to1D(2, 2, L)].TimeOnState = 5;
    h_population[to1D(2, 2, L)].AgeYears = 25;

    h_population[to1D(3, 3, L)].Health = H;
    h_population[to1D(3, 3, L)].Swap = H;
    h_population[to1D(3, 3, L)].StateTime = 5;
    h_population[to1D(3, 3, L)].TimeOnState = 5;
    h_population[to1D(3, 3, L)].AgeYears = 75;

    h_population[to1D(4, 4, L)].Health = H;
    h_population[to1D(4, 4, L)].Swap = H;
    h_population[to1D(4, 4, L)].Days = 365 * 85;
    h_population[to1D(4, 4, L)].AgeDeathDays = 365 * 80;

    // Copy to device
    cudaMemcpy(d_population, h_population, gridSize * sizeof(GPUPerson),
        cudaMemcpyHostToDevice);

    // Setup RNG
    setupRNG(gridSize);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (gridSize + blockSize - 1) / blockSize;

    H_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L,
        d_hospitalBeds, d_icuBeds);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    // Get final bed counts
    int finalBeds, finalICUBeds;
    cudaMemcpy(&finalBeds, d_hospitalBeds, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&finalICUBeds, d_icuBeds, sizeof(int), cudaMemcpyDeviceToHost);

    // Copy results back
    cudaMemcpy(h_population, d_population, gridSize * sizeof(GPUPerson),
        cudaMemcpyDeviceToHost);

    // Print results
    printf("\nFinal beds: Hospital=%d, ICU=%d\n", finalBeds, finalICUBeds);

    printf("Case 1 (young): Final state=%d\n", h_population[to1D(2, 2, L)].Swap);
    printf("Case 2 (elderly): Final state=%d\n", h_population[to1D(3, 3, L)].Swap);
    printf("Case 3 (natural death): Final state=%d\n", h_population[to1D(4, 4, L)].Swap);

    // Cleanup
    delete[] h_population;
    cudaFree(d_population);
    cudaFree(d_hospitalBeds);
    cudaFree(d_icuBeds);
    cleanupRNG();
}

int main() {
    // Initialize CUDA
    cudaFree(0);

    printf("Starting H_kernel test...\n");

    // Setup city
    setupCityParameters(ROC);

    // Setup GPU constants
    setupGPUConstants();

    // Run test
    test_H_kernel();

    // Cleanup
    cleanupGPUConstants();

    printf("Test completed!\n");
    return 0;
}