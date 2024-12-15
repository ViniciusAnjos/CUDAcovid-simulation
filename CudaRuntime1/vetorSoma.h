

// Função kernel que soma dois vetores na GPU
__global__ void somaVetoresCUDA(int* A, int* B, int* C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // Índice global do thread
    if (idx < N) {
        C[idx] = A[idx] + B[idx];  // Soma elemento por elemento
    }
}

// Função auxiliar que aloca memória na GPU e chama o kernel
void somarVetores(int* A, int* B, int* C, int N) {
    int* d_A, * d_B, * d_C;

    // Aloca memória para os vetores na GPU
    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));

    // Copia os vetores A e B da memória da CPU para a GPU
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice);

    // Definir o número de blocos e threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Chama o kernel na GPU
    somaVetoresCUDA << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N);

    // Aguarda a execução do kernel
    cudaDeviceSynchronize();

    // Copia o resultado de volta para a memória da CPU
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Libera a memória alocada na GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

