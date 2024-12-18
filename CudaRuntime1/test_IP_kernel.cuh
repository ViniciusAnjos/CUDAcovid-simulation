#include <stdio.h>
#include <cuda_runtime.h>
#include "gpu_define.cuh"
#include "gpu_person.cuh"
#include "gpu_utils.cuh"
#include "gpu_aleat.cuh"
#include "gpu_neighbors.cuh"

#include "IP_kernel.cuh"

void test_IP_state() {
    const int L = 5;
    const int gridDim = (L + 2) * (L + 2);  // Total grid size including boundaries

    // Allocate and initialize test population
    GPUPerson* h_population = new GPUPerson[gridDim];
    GPUPerson* d_population;
    cudaMalloc(&d_population, gridDim * sizeof(GPUPerson));

    // Initialize all as Susceptible
    printf("\nInitializing population...\n");
    for (int i = 0; i < gridDim; i++) {
        h_population[i].Health = S;
        h_population[i].Swap = S;
        h_population[i].TimeOnState = 0;
        h_population[i].StateTime = 0;
        h_population[i].Days = 0;
        h_population[i].AgeDeathDays = 365 * 80;  // 80 years
        h_population[i].Checked = 0;
        h_population[i].Exponent = 0;
    }

    // Set up test cases
    printf("\nSetting up test cases:\n");

    // Case 1: New IP case
    printf("\nCase 1 - New IP case (2,2):\n");
    h_population[to1D(2, 2, L)].Health = IP;
    h_population[to1D(2, 2, L)].StateTime = 10;
    h_population[to1D(2, 2, L)].TimeOnState = 0;
    printf("- StateTime: %d\n", h_population[to1D(2, 2, L)].StateTime);
    printf("- TimeOnState: %d\n", h_population[to1D(2, 2, L)].TimeOnState);
    printf("- Days: %d\n", h_population[to1D(2, 2, L)].Days);
    printf("- AgeDeathDays: %d\n", h_population[to1D(2, 2, L)].AgeDeathDays);

    // Case 2: IP about to transition
    printf("\nCase 2 - Ready to Transition (3,3):\n");
    h_population[to1D(3, 3, L)].Health = IP;
    h_population[to1D(3, 3, L)].StateTime = 5;
    h_population[to1D(3, 3, L)].TimeOnState = 5;
    printf("- StateTime: %d\n", h_population[to1D(3, 3, L)].StateTime);
    printf("- TimeOnState: %d\n", h_population[to1D(3, 3, L)].TimeOnState);
    printf("- Days: %d\n", h_population[to1D(3, 3, L)].Days);
    printf("- AgeDeathDays: %d\n", h_population[to1D(3, 3, L)].AgeDeathDays);

    // Case 3: IP about to die naturally
    printf("\nCase 3 - Natural Death Case (4,4):\n");
    h_population[to1D(4, 4, L)].Health = IP;
    h_population[to1D(4, 4, L)].Days = 365 * 85;  // 85 years
    h_population[to1D(4, 4, L)].AgeDeathDays = 365 * 80;  // Dies at 80 years
    h_population[to1D(4, 4, L)].StateTime = 5;  // Adding state time for completeness
    h_population[to1D(4, 4, L)].TimeOnState = 0;
    printf("- StateTime: %d\n", h_population[to1D(4, 4, L)].StateTime);
    printf("- TimeOnState: %d\n", h_population[to1D(4, 4, L)].TimeOnState);
    printf("- Days: %d\n", h_population[to1D(4, 4, L)].Days);
    printf("- AgeDeathDays: %d\n", h_population[to1D(4, 4, L)].AgeDeathDays);

    // Print transition probabilities
    double verify_probIPtoIA, verify_probLight, verify_probModerate;
    cudaMemcpyFromSymbol(&verify_probIPtoIA, d_ProbIPtoIA, sizeof(double));
    cudaMemcpyFromSymbol(&verify_probLight, d_ProbToBecomeISLight, sizeof(double));
    cudaMemcpyFromSymbol(&verify_probModerate, d_ProbToBecomeISModerate, sizeof(double));

    printf("\nTransition Probabilities:\n");
    printf("ProbIPtoIA: %f\n", verify_probIPtoIA);
    printf("ProbToBecomeISLight: %f\n", verify_probLight);
    printf("ProbToBecomeISModerate: %f\n", verify_probModerate);

    // Print initial grid
    printf("\nInitial Grid State:\n");
    printf("S: Susceptible, P: Pre-symptomatic\n");
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            printf("%c ", h_population[to1D(i, j, L)].Health == IP ? 'P' : 'S');
        }
        printf("\n");
    }

    // Copy to device and run kernel
    cudaMemcpy(d_population, h_population, gridDim * sizeof(GPUPerson),
        cudaMemcpyHostToDevice);

    setupRNG(gridDim);

    // Calculate kernel launch parameters
    int blockSize = 256;
    int numBlocks = (gridDim + blockSize - 1) / blockSize;
    printf("\nGrid dimensions: %d x %d = %d total cells\n", L + 2, L + 2, gridDim);
    printf("Launch configuration: %d blocks of %d threads each\n", numBlocks, blockSize);

    printf("\nLaunching IP_kernel with %d blocks, %d threads per block\n",
        numBlocks, blockSize);

    IP_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Get results
    cudaMemcpy(h_population, d_population, gridDim * sizeof(GPUPerson),
        cudaMemcpyDeviceToHost);

    // Print final state
    printf("\nFinal Grid State:\n");
    printf("S: Susceptible, P: Pre-symptomatic, A: Asymptomatic\n");
    printf("L: Light, M: Moderate, V: Severe, D: Dead\n");
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            char state = 'S';
            int idx = to1D(i, j, L);
            switch (h_population[idx].Swap) {
            case IP: state = 'P'; break;
            case IA: state = 'A'; break;
            case ISLight: state = 'L'; break;
            case ISModerate: state = 'M'; break;
            case ISSevere: state = 'V'; break;
            case Dead: state = 'D'; break;
            }
            printf("%c ", state);
        }
        printf("\n");
    }

    // Print detailed case results
    printf("\nDetailed Final State for Test Cases:\n");

    printf("\nCase 1 (2,2) Final State:\n");
    int idx = to1D(2, 2, L);
    printf("- Health: %d\n", h_population[idx].Health);
    printf("- Swap: %d\n", h_population[idx].Swap);
    printf("- TimeOnState: %d\n", h_population[idx].TimeOnState);
    printf("- StateTime: %d\n", h_population[idx].StateTime);
    printf("- Days: %d\n", h_population[idx].Days);
    printf("- AgeDeathDays: %d\n", h_population[idx].AgeDeathDays);

    printf("\nCase 2 (3,3) Final State:\n");
    idx = to1D(3, 3, L);
    printf("- Health: %d\n", h_population[idx].Health);
    printf("- Swap: %d\n", h_population[idx].Swap);
    printf("- TimeOnState: %d\n", h_population[idx].TimeOnState);
    printf("- StateTime: %d\n", h_population[idx].StateTime);
    printf("- Days: %d\n", h_population[idx].Days);
    printf("- AgeDeathDays: %d\n", h_population[idx].AgeDeathDays);

    printf("\nCase 3 (4,4) Final State:\n");
    idx = to1D(4, 4, L);
    printf("- Health: %d\n", h_population[idx].Health);
    printf("- Swap: %d\n", h_population[idx].Swap);
    printf("- TimeOnState: %d\n", h_population[idx].TimeOnState);
    printf("- StateTime: %d\n", h_population[idx].StateTime);
    printf("- Days: %d\n", h_population[idx].Days);
    printf("- AgeDeathDays: %d\n", h_population[idx].AgeDeathDays);

    // Cleanup
    delete[] h_population;
    cudaFree(d_population);
    cleanupRNG();
}

int main() {
    int city = SP;
    setupCityParameters(city);
    setupGPUConstants();

    test_IP_state();

    cleanupGPUConstants();
    return 0;
}