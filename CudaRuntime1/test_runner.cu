// test_runner.cu
// Entry point for all unit tests.
// Compile and run this instead of covid.cu to verify kernel correctness.
//
// nvcc test_runner.cu -o test_runner && ./test_runner

#include <stdio.h>
#include <cuda_runtime.h>

#include "gpu_define.cuh"
#include "gpu_person.cuh"
#include "gpu_utils.cuh"
#include "gpu_aleat.cuh"
#include "Update_kernel.cuh"

#include "test_update_kernel.cuh"

int main() {
    // Print GPU info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s\n\n", prop.name);

    // Set up minimal constants required by the kernels under test
    int city = SP;
    setupCityParameters(city);
    setupGPUConstants();

    int exitCode = runUpdateKernelTests();

    cleanupGPUConstants();
    return exitCode;
}
