#include <stdio.h>
#include <cuda_runtime.h>
#include "gpu_define.cuh"
#include "gpu_person.cuh"
#include "gpu_utils.cuh"
#include "gpu_update_boundaries.cuh"

int main() {
    // Test with a small grid
    const int L = 4;  // 4x4 grid for easier visualization
    const int gridSize = (L + 2) * (L + 2);  // Including boundaries

    // Allocate host memory
    GPUPerson* h_population = new GPUPerson[gridSize];

    printf("Testing boundary conditions with %dx%d grid\n", L, L);

    // Initialize with distinct health states for testing
    for (int i = 0; i <= L + 1; i++) {
        for (int j = 0; j <= L + 1; j++) {
            int idx = to1D(i, j, L);

            // Default initialization
            h_population[idx].Health = 0;

            // Set inner grid cells with unique values
            if (i >= 1 && i <= L && j >= 1 && j <= L) {
                // Use different values for each cell to clearly see the boundary effect
                h_population[idx].Health = (i - 1) * L + (j - 1) + 1;
            }
        }
    }

    // Print initial state
    printf("\nInitial Grid (0 = boundary, numbers = inner cells):\n");
    for (int i = 0; i <= L + 1; i++) {
        for (int j = 0; j <= L + 1; j++) {
            printf("%2d ", h_population[to1D(i, j, L)].Health);
        }
        printf("\n");
    }

    // Allocate device memory
    GPUPerson* d_population;
    cudaMalloc(&d_population, gridSize * sizeof(GPUPerson));

    // Copy to device
    cudaMemcpy(d_population, h_population, gridSize * sizeof(GPUPerson), cudaMemcpyHostToDevice);

    // Launch kernel with L threads (enough for boundaries)
    int blockSize = 128;
    int numBlocks = (L + blockSize - 1) / blockSize;
    printf("\nLaunching updateBoundaries_kernel with %d blocks of %d threads\n",
        numBlocks, blockSize);

    updateBoundaries_kernel << <numBlocks, blockSize >> > (d_population, L);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy back to host
    cudaMemcpy(h_population, d_population, gridSize * sizeof(GPUPerson), cudaMemcpyDeviceToHost);

    // Print final state
    printf("\nAfter Boundary Update:\n");
    for (int i = 0; i <= L + 1; i++) {
        for (int j = 0; j <= L + 1; j++) {
            printf("%2d ", h_population[to1D(i, j, L)].Health);
        }
        printf("\n");
    }

    // Verify a few boundary conditions
    printf("\nVerification:\n");
    bool allCorrect = true;

    // Check top boundary
    for (int j = 1; j <= L; j++) {
        if (h_population[to1D(0, j, L)].Health != h_population[to1D(L, j, L)].Health) {
            printf("ERROR: Top boundary mismatch at j=%d\n", j);
            allCorrect = false;
        }
    }

    // Check bottom boundary
    for (int j = 1; j <= L; j++) {
        if (h_population[to1D(L + 1, j, L)].Health != h_population[to1D(1, j, L)].Health) {
            printf("ERROR: Bottom boundary mismatch at j=%d\n", j);
            allCorrect = false;
        }
    }

    // Check corners
    if (h_population[to1D(0, 0, L)].Health != h_population[to1D(L, L, L)].Health) {
        printf("ERROR: Top-left corner mismatch\n");
        allCorrect = false;
    }

    if (allCorrect) {
        printf("All boundary checks PASSED!\n");
    }
    else {
        printf("Some boundary checks FAILED!\n");
    }

    // Clean up
    delete[] h_population;
    cudaFree(d_population);

    return 0;
}