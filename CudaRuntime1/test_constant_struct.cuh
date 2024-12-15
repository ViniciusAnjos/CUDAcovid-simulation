#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>


void checkCudaError(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(error));
        exit(-1);
    }
}

#include "gpu_define.cuh"
#include "gpu_person.cuh"

__global__ void testKernel(GPUPerson* d_person, double* probNaturalDeath,
    double* probRecoveryH, double* ProbBirthAge, int* AgeMin) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        printf("\nTesting GPU Constants:\n");
        printf("Grid size (L): %d\n", d_L);
        printf("Population (N): %d\n", d_N);
        printf("Beta: %f\n", d_Beta);
        printf("Days: %d\n", d_DAYS);

        printf("\nDisease States:\n");
        printf("Susceptible (S): %d\n", d_S);
        printf("Exposed (E): %d\n", d_E);
        printf("Pre-infectious (IP): %d\n", d_IP);

        printf("\nProbabilities for different ages:\n");
        printf("Natural Death Prob (age 30): %f\n", probNaturalDeath[30]);
        printf("Natural Death Prob (age 60): %f\n", probNaturalDeath[60]);
        printf("Recovery H Prob (age 30): %f\n", probRecoveryH[30]);
        printf("Recovery H Prob (age 60): %f\n", probRecoveryH[60]);

        printf("\nTime Constants:\n");
        printf("MinLatency: %f\n", d_MinLatency);
        printf("MaxLatency: %f\n", d_MaxLatency);

        printf("prob birth age[17]: %f\n", ProbBirthAge[17]);
        printf("prob birth age[8]: %f\n", ProbBirthAge[8]);
        printf("age min[17]: %d\n", AgeMin[17]);
        printf("age min[8]: %d\n", AgeMin[8]);

    }
}

int main() {
    printf("Starting GPU Constants and Structure Test\n");

    // Initialize probability arrays first
    double h_ProbNaturalDeath[121] = { 0 };
    double h_ProbRecoveryModerate[121] = { 0 };
    double h_ProbRecoverySevere[121] = { 0 };
    double h_ProbRecoveryH[121] = { 0 };
    double h_ProbRecoveryICU[121] = { 0 };

    double h_ProbBirthAge[21] = { 0 };
    double h_SumProbBirthAge[21] = { 0 };
    int h_AgeMin[21] = { 0 };
    int h_AgeMax[21] = { 0 };

    // Debug: Print values before GPU transfer
    buildArrays(h_ProbNaturalDeath, h_ProbRecoveryModerate, h_ProbRecoverySevere,
        h_ProbRecoveryH, h_ProbRecoveryICU, h_ProbBirthAge, h_SumProbBirthAge, h_AgeMin, h_AgeMax);

    printf("\nCPU Values before transfer:\n");
    printf("Recovery H Prob (age 30): %f\n", h_ProbRecoveryH[30]);
    printf("Recovery H Prob (age 60): %f\n", h_ProbRecoveryH[60]);

    // Initialize GPU constants
    setupGPUConstants();

    // Allocate test array
    GPUPerson* d_person;
    allocateGPUMemory(&d_person, L);

    // Launch kernel with the array pointers
    int blockSize = 256;
    int numBlocks = (L * L + blockSize - 1) / blockSize;
    testKernel << <numBlocks, blockSize >> > (d_person, d_ProbNaturalDeath, d_ProbRecoveryH, d_ProbBirthAge, d_AgeMax);

    // Wait and check for errors
    cudaError_t error = cudaDeviceSynchronize();
    checkCudaError(error, "Kernel execution failed");

    // Free memory
    freeGPUMemory(d_person);
    cleanupGPUConstants();

    printf("\nTest completed successfully!\n");
    return 0;
}