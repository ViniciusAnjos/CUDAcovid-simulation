

// Device pointer to store RNG states for all threads
__device__ unsigned int* d_rngStates;

// The random number generation function - matches the original aleat() implementation
__device__ double generateRandom(unsigned int* state) {
    // We use unsigned long long for the multiplication to prevent overflow issues
    unsigned long long temp = (unsigned long long) * state * 888121ULL;

    // Keep only 32 bits to match CPU behavior, exactly like original aleat()
    *state = (unsigned int)(temp & 0xFFFFFFFFULL);

    // Convert to double and divide by MAXNUM, matching original implementation
    return (double)*state / 4294967295.0;
}

// Kernel to initialize the RNG states for each thread
__global__ void initRNG(unsigned int* states, unsigned int seed, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Each thread gets a unique seed derived from the initial seed
        states[idx] = seed * (idx + 1);
    }
}

// Host function to set up the RNG system
__host__ void setupRNG(int size) {
    // Allocate memory for RNG states on the GPU
    cudaMalloc((void**)&d_rngStates, size * sizeof(unsigned int));

    // Calculate dimensions for the initialization kernel
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    // Initialize all RNG states using the same seed as your original code
    initRNG << <numBlocks, blockSize >> > (d_rngStates, 893221891, size);
    cudaDeviceSynchronize();
}

// Host function to clean up RNG resources
__host__ void cleanupRNG() {
    cudaFree(d_rngStates);
}

