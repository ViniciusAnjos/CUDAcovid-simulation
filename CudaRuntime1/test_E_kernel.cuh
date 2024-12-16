#include <stdio.h>
#include <cuda_runtime.h>
#include "gpu_define.cuh"
#include "gpu_person.cuh"
#include "gpu_utils.cuh"
#include "gpu_aleat.cuh"

#include "E_kernel.cuh"

void test_E_state() {
    const int L = 5;
    const int gridSize = (L + 2) * (L + 2);

    // Verify device constants first
    printf("\nVerifying device constants:\n");
    int verify_E, verify_IP, verify_Dead;
    cudaError_t err;

    err = cudaMemcpyFromSymbol(&verify_E, d_E, sizeof(int));
    if (err != cudaSuccess) {
        printf("Error reading d_E constant: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMemcpyFromSymbol(&verify_IP, d_IP, sizeof(int));
    if (err != cudaSuccess) {
        printf("Error reading d_IP constant: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMemcpyFromSymbol(&verify_Dead, d_Dead, sizeof(int));
    if (err != cudaSuccess) {
        printf("Error reading d_Dead constant: %s\n", cudaGetErrorString(err));
        return;
    }

    printf("Device Constants:\n");
    printf("d_E = %d\n", verify_E);
    printf("d_IP = %d\n", verify_IP);
    printf("d_Dead = %d\n", verify_Dead);

    // Allocate and initialize test population
    GPUPerson* h_population = new GPUPerson[gridSize];
    GPUPerson* d_population;
    err = cudaMalloc(&d_population, gridSize * sizeof(GPUPerson));
    if (err != cudaSuccess) {
        printf("Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        return;
    }

    printf("\nInitializing population...\n");
    for (int i = 0; i < gridSize; i++) {
        h_population[i].Health = S;
        h_population[i].Swap = S;
        h_population[i].TimeOnState = 0;
        h_population[i].StateTime = 0;
        h_population[i].Days = 0;
        h_population[i].AgeDeathDays = 365 * 80;  // 80 years
    }

    printf("\nSetting up test cases:\n");

    // Case 1: Just became Exposed
    printf("\nCase 1 - New Exposed (2,2):\n");
    h_population[to1D(2, 2, L)].Health = E;
    h_population[to1D(2, 2, L)].StateTime = 10;
    printf("- Health: %d (E)\n", h_population[to1D(2, 2, L)].Health);
    printf("- TimeOnState: %d\n", h_population[to1D(2, 2, L)].TimeOnState);
    printf("- StateTime: %d\n", h_population[to1D(2, 2, L)].StateTime);
    printf("- Days: %d\n", h_population[to1D(2, 2, L)].Days);
    printf("- AgeDeathDays: %d\n", h_population[to1D(2, 2, L)].AgeDeathDays);

    // Case 2: About to transition
    printf("\nCase 2 - Ready to Transition (3,3):\n");
    h_population[to1D(3, 3, L)].Health = E;
    h_population[to1D(3, 3, L)].StateTime = 5;
    h_population[to1D(3, 3, L)].TimeOnState = 5;
    printf("- Health: %d (E)\n", h_population[to1D(3, 3, L)].Health);
    printf("- TimeOnState: %d\n", h_population[to1D(3, 3, L)].TimeOnState);
    printf("- StateTime: %d\n", h_population[to1D(3, 3, L)].StateTime);
    printf("- Days: %d\n", h_population[to1D(3, 3, L)].Days);
    printf("- AgeDeathDays: %d\n", h_population[to1D(3, 3, L)].AgeDeathDays);

    // Case 3: Natural death case
    printf("\nCase 3 - Natural Death (4,4):\n");
    h_population[to1D(4, 4, L)].Health = E;
    h_population[to1D(4, 4, L)].Days = 365 * 85;  // 85 years
    printf("- Health: %d (E)\n", h_population[to1D(4, 4, L)].Health);
    printf("- TimeOnState: %d\n", h_population[to1D(4, 4, L)].TimeOnState);
    printf("- StateTime: %d\n", h_population[to1D(4, 4, L)].StateTime);
    printf("- Days: %d\n", h_population[to1D(4, 4, L)].Days);
    printf("- AgeDeathDays: %d\n", h_population[to1D(4, 4, L)].AgeDeathDays);

    // Print initial grid state
    printf("\nInitial Grid State:\n");
    printf("S: Susceptible, E: Exposed\n");
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            printf("%c ", h_population[to1D(i, j, L)].Health == E ? 'E' : 'S');
        }
        printf("\n");
    }

    // Copy to device
    err = cudaMemcpy(d_population, h_population, gridSize * sizeof(GPUPerson),
        cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Failed to copy to device: %s\n", cudaGetErrorString(err));
        return;
    }

    // Setup RNG
    setupRNG(gridSize);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (gridSize + blockSize - 1) / blockSize;
    printf("\nLaunching E_kernel with %d blocks, %d threads per block\n",
        numBlocks, blockSize);

    E_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);

    // Check for kernel launch error
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // Get results back
    err = cudaMemcpy(h_population, d_population, gridSize * sizeof(GPUPerson),
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Failed to copy from device: %s\n", cudaGetErrorString(err));
        return;
    }

    // Print final state
    printf("\nFinal Grid State and Details:\n");
    printf("S: Susceptible, E: Exposed, I: Infectious, D: Dead\n");
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            char state = 'S';
            int idx = to1D(i, j, L);
            if (h_population[idx].Swap == E) state = 'E';
            else if (h_population[idx].Swap == IP) state = 'I';
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
    printf("- TimeOnState: %d\n", h_population[to1D(2, 2, L)].TimeOnState);
    printf("- StateTime: %d\n", h_population[to1D(2, 2, L)].StateTime);
    printf("- Days: %d\n", h_population[to1D(2, 2, L)].Days);

    printf("\nCase 2 (3,3) Final State:\n");
    printf("- Health: %d\n", h_population[to1D(3, 3, L)].Health);
    printf("- Swap: %d\n", h_population[to1D(3, 3, L)].Swap);
    printf("- TimeOnState: %d\n", h_population[to1D(3, 3, L)].TimeOnState);
    printf("- StateTime: %d\n", h_population[to1D(3, 3, L)].StateTime);
    printf("- Days: %d\n", h_population[to1D(3, 3, L)].Days);

    printf("\nCase 3 (4,4) Final State:\n");
    printf("- Health: %d\n", h_population[to1D(4, 4, L)].Health);
    printf("- Swap: %d\n", h_population[to1D(4, 4, L)].Swap);
    printf("- TimeOnState: %d\n", h_population[to1D(4, 4, L)].TimeOnState);
    printf("- StateTime: %d\n", h_population[to1D(4, 4, L)].StateTime);
    printf("- Days: %d\n", h_population[to1D(4, 4, L)].Days);
    printf("- AgeDeathDays: %d\n", h_population[to1D(4, 4, L)].AgeDeathDays);

    // Cleanup
    delete[] h_population;
    cudaFree(d_population);
    cleanupRNG();
}

int main() {
    int city = ROC;  // Use Rocinha parameters
    setupCityParameters(city);
    setupGPUConstants();

    // Run the test
    test_E_state();

    // Cleanup GPU constants
    cleanupGPUConstants();

    return 0;
}