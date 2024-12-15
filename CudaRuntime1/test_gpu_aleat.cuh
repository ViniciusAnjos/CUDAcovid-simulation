#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "gpu_aleat.cuh"

// This function helps us catch and handle CUDA errors properly
void checkCudaError(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(error));
        exit(-1);
    }
}

// This kernel generates random numbers using our GPU implementation
// It handles both the state management and number generation for each thread
__global__ void generateTestNumbers(double* numbers, unsigned int* states, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // Each thread gets its own RNG state
        unsigned int localState = states[idx];

        // Generate a random number using our implementation
        numbers[idx] = gpuAleatMatch(&localState);

        // Save the updated state for next time
        states[idx] = localState;
    }
}

// This function performs a comprehensive statistical analysis of our random numbers
void analyzeRandomNumbers(double* numbers, int count) {
    printf("\nStarting Statistical Analysis...\n");

    // First, let's look at a few numbers to verify basic generation
    printf("Sample of first 5 generated numbers:\n");
    for (int i = 0; i < 5; i++) {
        printf("Number %d: %.6f\n", i + 1, numbers[i]);
    }

    // Calculate basic statistical measures
    double sum = 0.0;
    double sumSquared = 0.0;
    double min = 1.0;
    double max = 0.0;
    int outOfRange = 0;

    // First pass through the data to gather basic statistics
    for (int i = 0; i < count; i++) {
        double num = numbers[i];

        // Check if numbers are in valid range
        if (num < 0.0 || num > 1.0) {
            outOfRange++;
        }

        // Track minimum and maximum values
        if (num < min) min = num;
        if (num > max) max = num;

        // Accumulate sums for mean and variance calculation
        sum += num;
        sumSquared += num * num;
    }

    // Calculate key statistical measures
    double mean = sum / count;
    double variance = (sumSquared / count) - (mean * mean);
    double stdDev = sqrt(variance);

    // Print comprehensive analysis results
    printf("\nDetailed Statistical Analysis\n");
    printf("============================\n");

    printf("\nBasic Properties:\n");
    printf("- Total numbers analyzed: %d\n", count);
    printf("- Numbers outside [0,1] range: %d\n", outOfRange);
    printf("- Value range: [%.10f to %.10f]\n", min, max);

    printf("\nStatistical Measures:\n");
    printf("- Mean: %.10f (Expected: 0.5000000000)\n", mean);
    printf("- Standard Deviation: %.10f (Expected: 0.2886751346)\n", stdDev);

    // Analyze distribution using bins
    const int numBins = 10;
    int bins[10] = { 0 };

    // Sort numbers into bins to check distribution
    for (int i = 0; i < count; i++) {
        int bin = (int)(numbers[i] * numBins);
        if (bin == numBins) bin = numBins - 1;  // Handle edge case of 1.0
        bins[bin]++;
    }

    // Print distribution analysis
    printf("\nDistribution Analysis:\n");
    printf("Expected numbers per bin: %d\n", count / numBins);
    for (int i = 0; i < numBins; i++) {
        double percentage = (bins[i] * 100.0) / count;
        printf("Bin [%.1f-%.1f): %7d numbers (%.2f%%)\n",
            i / 10.0, (i + 1) / 10.0, bins[i], percentage);
    }

    // Evaluate the quality of our RNG
    printf("\nQuality Assessment:\n");
    double meanError = fabs(mean - 0.5);
    double stdDevError = fabs(stdDev - 0.2886751346);
    const double errorThreshold = 0.01;  // 1% error tolerance

    printf("- Mean Error: %.10f\n", meanError);
    printf("- StdDev Error: %.10f\n", stdDevError);

    // Final assessment
    if (meanError < errorThreshold && stdDevError < errorThreshold) {
        printf("\nFINAL RESULT: PASSED ✓\n");
        printf("Statistical properties match expected values within tolerance.\n");
    }
    else {
        printf("\nFINAL RESULT: FAILED ✗\n");
        printf("Statistical properties deviate from expected values.\n");
    }
}

int main() {
    const int testSize = 1000000;  // We'll test with a million numbers for good statistical significance
    printf("Starting Random Number Generator Test\n");
    printf("Generating %d random numbers...\n", testSize);

    // Keep track of any CUDA errors
    cudaError_t error;

    // Step 1: Allocate memory for RNG states
    unsigned int* d_states;
    error = cudaMalloc(&d_states, testSize * sizeof(unsigned int));
    checkCudaError(error, "Failed to allocate device memory for RNG states");

    // Step 2: Initialize the RNG states
    int blockSize = 256;
    int numBlocks = (testSize + blockSize - 1) / blockSize;
    initRNG << <numBlocks, blockSize >> > (d_states, 893221891, testSize);
    error = cudaGetLastError();
    checkCudaError(error, "Failed to initialize RNG states");

    // Step 3: Allocate memory for the random numbers
    double* d_numbers;
    error = cudaMalloc(&d_numbers, testSize * sizeof(double));
    checkCudaError(error, "Failed to allocate device memory for numbers");

    // Step 4: Generate the random numbers
    generateTestNumbers << <numBlocks, blockSize >> > (d_numbers, d_states, testSize);
    error = cudaGetLastError();
    checkCudaError(error, "Failed to generate random numbers");

    // Step 5: Allocate host memory and copy results back
    double* h_numbers = (double*)malloc(testSize * sizeof(double));
    if (h_numbers == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        cudaFree(d_numbers);
        cudaFree(d_states);
        exit(-1);
    }

    // Copy the generated numbers back to host for analysis
    error = cudaMemcpy(h_numbers, d_numbers, testSize * sizeof(double),
        cudaMemcpyDeviceToHost);
    checkCudaError(error, "Failed to copy results back to host");

    // Step 6: Analyze the results
    analyzeRandomNumbers(h_numbers, testSize);

    // Step 7: Clean up all allocated memory
    free(h_numbers);
    cudaFree(d_numbers);
    cudaFree(d_states);

    printf("\nTest completed successfully!\n");
    return 0;
}