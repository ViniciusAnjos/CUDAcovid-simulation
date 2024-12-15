#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gpu_define.cuh"
#include "gpu_utils.cuh"

//HAHAHAHAHAHAHAHAH


// Test kernel to verify grid conversion and neighbor access
__global__ void testGridConversion(int L) {
    // Test a few specific positions
    if (threadIdx.x == 0) {
        printf("\nTesting Grid Conversions for %dx%d grid:\n", L, L);

        // Test cases for different positions
        int test_positions[][2] = {
            {1, 1},    // Corner
            {1, L},    // Top right
            {L, 1},    // Bottom left
            {L, L},    // Bottom right
            {L / 2, L / 2} // Center
        };

        for (int t = 0; t < 5; t++) {
            int i = test_positions[t][0];
            int j = test_positions[t][1];

            // Convert to 1D and back
            int idx = to1D(i, j, L);
            int back_i, back_j;
            to2D(idx, L, back_i, back_j);

            printf("\nPosition (%d,%d):\n", i, j);
            printf("1D index: %d\n", idx);
            printf("Convert back: (%d,%d)\n", back_i, back_j);

            // Test neighbor access
            printf("Neighbors:\n");
            int neighborIndices[8];
            int numNeighbors;
            getNeighborIndices(i, j, L, HIGH, neighborIndices, numNeighbors);

            printf("HIGH density neighbors:\n");
            for (int n = 0; n < numNeighbors; n++) {
                int ni, nj;
                to2D(neighborIndices[n], L, ni, nj);
                printf("  Neighbor %d: (%d,%d) at index %d\n",
                    n + 1, ni, nj, neighborIndices[n]);
            }

            // Test 4-neighbor pattern
            getNeighborIndices(i, j, L, LOW, neighborIndices, numNeighbors);
            printf("LOW density neighbors:\n");
            for (int n = 0; n < numNeighbors; n++) {
                int ni, nj;
                to2D(neighborIndices[n], L, ni, nj);
                printf("  Neighbor %d: (%d,%d) at index %d\n",
                    n + 1, ni, nj, neighborIndices[n]);
            }

            printf("\n-------------------\n");
        }
    }
}

// Function to visualize a small section of the grid
void visualizeGrid(int i, int j, int L) {
    printf("\nVisualizing 3x3 section around (%d,%d):\n", i, j);
    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            int ni = i + di;
            int nj = j + dj;
            if (ni <= 0) ni = L;
            if (ni > L) ni = 1;
            if (nj <= 0) nj = L;
            if (nj > L) nj = 1;

            if (di == 0 && dj == 0)
                printf("[%2d,%2d]", ni, nj);
            else
                printf("(%2d,%2d)", ni, nj);
        }
        printf("\n");
    }
}

int main() {
    // Test with a small grid first
    int L = 4;  // 4x4 grid for easy visualization
    printf("Starting Grid Conversion and Neighbor Test\n");

    // Launch test kernel
    testGridConversion << <1, 1 >> > (L);
    cudaDeviceSynchronize();

    // CPU-side visualization
    printf("\nCPU-side Grid Visualization:\n");
    visualizeGrid(1, 1, L);    // Corner
    visualizeGrid(2, 2, L);    // Inner position
    visualizeGrid(4, 4, L);    // Corner

    // Test with actual grid size
    L = 3200;
    printf("\nTesting with full grid size (L=%d):\n", L);
    testGridConversion << <1, 1 >> > (L);
    cudaDeviceSynchronize();

    printf("\nTest completed!\n");
    return 0;
}