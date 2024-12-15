#ifndef GPU_UTILS_CUH
#define GPU_UTILS_CUH

// Convert 2D grid position to 1D array index
__host__ __device__ inline int to1D(int i, int j, int L) {
    return i * (L + 2) + j;
}

// Convert 1D array index to 2D grid position
__host__ __device__ inline void to2D(int idx, int L, int& i, int& j) {
    i = idx / (L + 2);
    j = idx % (L + 2);
}

// Handle periodic boundary wrapping
__device__ inline int wrapIndex(int idx, int L) {
    if (idx <= 0) return L;        // If below min, wrap to max
    if (idx > L) return 1;         // If above max, wrap to min
    return idx;
}

// Get index of neighbor considering periodic boundaries
__device__ inline int getNeighborIndex(int i, int j, int di, int dj, int L) {
    // Wrap i and j coordinates
    int ni = wrapIndex(i + di, L);
    int nj = wrapIndex(j + dj, L);

    // Convert to 1D index
    return to1D(ni, nj, L);
}

// Get all neighbor indices based on density
__device__ inline void getNeighborIndices(int i, int j, int L, int density,
    int* neighborIndices, int& numNeighbors) {
    if (density == HIGH) {  // 8 neighbors
        numNeighbors = 8;
        int idx = 0;
        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                if (di != 0 || dj != 0) {  // Skip central cell
                    neighborIndices[idx++] = getNeighborIndex(i, j, di, dj, L);
                }
            }
        }
    }
    else {  // 4 neighbors
        numNeighbors = 4;
        neighborIndices[0] = getNeighborIndex(i, j, -1, 0, L);  // North
        neighborIndices[1] = getNeighborIndex(i, j, 1, 0, L);   // South
        neighborIndices[2] = getNeighborIndex(i, j, 0, -1, L);  // West
        neighborIndices[3] = getNeighborIndex(i, j, 0, 1, L);   // East
    }
}

// Check if index is within valid boundaries
__device__ inline bool isValidIndex(int i, int j, int L) {
    return (i > 0 && i <= L && j > 0 && j <= L);
}

#endif