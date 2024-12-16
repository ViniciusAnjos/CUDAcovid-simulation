

#include<cuda_runtime.h>

// CPU structure
struct Individual {
    int Health;
    int Swap;
    int Gender;
    int AgeYears;
    int AgeDays;
    int AgeDeathYears;
    int AgeDeathDays;
    int StateTime;
    int TimeOnState;
    int Days;
    int Isolation;
    int Exponent;
    int Checked;
};

struct __align__(16) GPUPerson {
    int Health;
    int Swap;
    int Gender;
    int AgeYears;
    int AgeDays;
    int AgeDeathYears;
    int AgeDeathDays;
    int StateTime;
    int TimeOnState;
    int Days;
    int Isolation;
    int Exponent;
    int Checked;
};

// Memory management functions
__host__ void allocateGPUMemory(GPUPerson** d_person, int L) {
    cudaError_t err = cudaMalloc((void**)d_person, (L + 2) * (L + 2) * sizeof(GPUPerson));
    checkCudaError(err, "Failed to allocate device memory");
}

__host__ void freeGPUMemory(GPUPerson* d_person) {
    cudaError_t err = cudaFree(d_person);
    checkCudaError(err, "Failed to free device memory");
}

__host__ void copyToGPU(GPUPerson* d_person, Individual* h_person, int L) {
    cudaError_t err = cudaMemcpy(d_person, h_person, (L + 2) * (L + 2) * sizeof(GPUPerson),
        cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy to device");
}

__host__ void copyFromGPU(Individual* h_person, GPUPerson* d_person, int L) {
    cudaError_t err = cudaMemcpy(h_person, d_person, (L + 2) * (L + 2) * sizeof(GPUPerson),
        cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy from device");
}



