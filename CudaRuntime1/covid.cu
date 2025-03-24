#include <stdio.h>
#include <cuda_runtime.h>
#include "gpu_define.cuh"
#include "gpu_utils.cuh"

// Test kernel to check if global device pointer is accessible
__global__ void test_global_pointer_kernel(double* output, int size) {
    int idx = threadIdx.x;
    if (idx < size) {
        // Try to read from the global device pointer
        output[idx] = d_ProbRecoveryH[idx];
    }
}

int main() {
    // Set up GPU constants
    setupGPUConstants();

    // Allocate output array
    double* h_output = new double[121];
    double* d_output;
    cudaMalloc(&d_output, 121 * sizeof(double));

    // Launch test kernel
    test_global_pointer_kernel << <1, 121 >> > (d_output, 121);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(h_output, d_output, 121 * sizeof(double), cudaMemcpyDeviceToHost);

    // Print some values
    printf("Recovery probabilities from global device pointer:\n");
    printf("Age 0: %f\n", h_output[0]);
    printf("Age 25: %f\n", h_output[25]);
    printf("Age 50: %f\n", h_output[50]);
    printf("Age 75: %f\n", h_output[75]);
    printf("Age 85: %f\n", h_output[85]);

    // Clean up
    delete[] h_output;
    cudaFree(d_output);
    cleanupGPUConstants();

    return 0;
}