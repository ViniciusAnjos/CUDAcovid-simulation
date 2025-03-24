__global__ void updateBoundaries_kernel(GPUPerson* population, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Handle horizontal boundaries (top and bottom rows)
    if (idx < L) {
        int j = idx + 1; // Skip the corners

        // Copy bottom row to top boundary
        population[to1D(0, j, L)].Health = population[to1D(L, j, L)].Health;

        // Copy top row to bottom boundary
        population[to1D(L + 1, j, L)].Health = population[to1D(1, j, L)].Health;
    }

    // Handle vertical boundaries (leftmost and rightmost columns)
    if (idx < L) {
        int i = idx + 1; // Skip the corners

        // Copy rightmost column to left boundary
        population[to1D(i, 0, L)].Health = population[to1D(i, L, L)].Health;

        // Copy leftmost column to right boundary
        population[to1D(i, L + 1, L)].Health = population[to1D(i, 1, L)].Health;
    }

    // Handle corners (one thread is sufficient)
    if (idx == 0) {
        population[to1D(0, 0, L)].Health = population[to1D(L, L, L)].Health;
        population[to1D(0, L + 1, L)].Health = population[to1D(L, 1, L)].Health;
        population[to1D(L + 1, 0, L)].Health = population[to1D(1, L, L)].Health;
        population[to1D(L + 1, L + 1, L)].Health = population[to1D(1, 1, L)].Health;
    }
}