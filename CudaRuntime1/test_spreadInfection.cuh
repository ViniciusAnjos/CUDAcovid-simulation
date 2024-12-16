#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "gpu_define.cuh"
#include "gpu_person.cuh"
#include "gpu_utils.cuh"
#include "gpu_aleat.cuh"
#include "gpu_neighbors.cuh"


__device__ void spreadInfection_kernel(int i, int j, GPUPerson* population,
    unsigned int* rngState, int L) {
    double rn = generateRandom(rngState);

    printf("Thread (%d,%d) rn for contacts = %f\n", i, j, rn);

    // Debug the random contacts calculation
    double range = d_MaxRandomContacts - d_MinRandomContacts;
    double scaled = rn * range;
    double final = scaled + d_MinRandomContacts;
    int RandomContacts = (int)final;

    printf("Thread (%d,%d) contact calculation: range=%f, scaled=%f, final=%f\n",
        i, j, range, scaled, final);
    printf("Thread (%d,%d) generating %d contacts\n", i, j, RandomContacts);

    for (int contact = 0; contact < RandomContacts; contact++) {
        int Randomi, Randomj;
        do {
            rn = generateRandom(rngState);
            Randomi = (int)(rn * L) + 1;

            rn = generateRandom(rngState);
            Randomj = (int)(rn * L) + 1;
        } while (Randomi == i && Randomj == j);

        int randomIdx = to1D(Randomi, Randomj, L);
        // Debug: Print random contact selection
        printf("Thread (%d,%d) checking contact at (%d,%d)\n", i, j, Randomi, Randomj);

        if (population[randomIdx].Health == d_S) {
            // Debug: Print exposure attempt
            printf("Thread (%d,%d) attempting exposure at (%d,%d)\n", i, j, Randomi, Randomj);

            int oldval = atomicAdd((int*)&population[randomIdx].Exponent, 1);

            // Debug: Print atomic operation result
            printf("Thread (%d,%d) exposure count at (%d,%d): %d\n",
                i, j, Randomi, Randomj, oldval + 1);

            if (population[randomIdx].Checked == 0 && oldval == 0) {
                double ProbContagion = 1.0 - pow(1.0 - d_Beta, (double)(oldval + 1));

                rn = generateRandom(rngState);
                // Debug: Print infection attempt
                printf("Thread (%d,%d) infection attempt at (%d,%d): prob=%f, rn=%f\n",
                    i, j, Randomi, Randomj, ProbContagion, rn);

                if (rn <= ProbContagion) {
                    population[randomIdx].Checked = 1;
                    population[randomIdx].Swap = d_E;
                    population[randomIdx].TimeOnState = 0;

                    rn = generateRandom(rngState);
                    population[randomIdx].StateTime = rn * (d_MaxLatency - d_MinLatency) + d_MinLatency;

                    // Debug: Print successful infection
                    printf("Thread (%d,%d) successfully infected (%d,%d)\n", i, j, Randomi, Randomj);
                }
            }
        }
    }
}

__global__ void test_spread_infection_kernel(GPUPerson* population,
    unsigned int* rngStates,
    int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * L) return;

    int i, j;
    to2D(idx, L, i, j);

    if (i > 0 && i <= L && j > 0 && j <= L) {
        if (isInfectious(population[to1D(i, j, L)].Health)) {
            unsigned int* myRNG = &rngStates[idx];
            spreadInfection_kernel(i, j, population, myRNG, L);
        }
    }
}

void test_spread_infection() {
    const int L = 10;
    const int gridSize = (L + 2) * (L + 2);

    GPUPerson* h_population = new GPUPerson[gridSize];
    GPUPerson* d_population;
    cudaMalloc(&d_population, gridSize * sizeof(GPUPerson));

    printConstants();

    for (int i = 0; i < gridSize; i++) {
        h_population[i].Health = S;
        h_population[i].Swap = S;
        h_population[i].Exponent = 0;
        h_population[i].Checked = 0;
        h_population[i].TimeOnState = 0;
        h_population[i].StateTime = 0;
    }

    h_population[to1D(2, 2, L)].Health = IP;
    h_population[to1D(5, 5, L)].Health = IA;
    h_population[to1D(8, 8, L)].Health = ISLight;

    printf("\nInitial Grid State:\n");
    printf("S: Susceptible, I: Infectious\n");
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            printf("%c ", h_population[to1D(i, j, L)].Health == S ? 'S' : 'I');
        }
        printf("\n");
    }

    cudaMemcpy(d_population, h_population, gridSize * sizeof(GPUPerson),
        cudaMemcpyHostToDevice);

    setupRNG(gridSize);

    int blockSize = 256;
    int numBlocks = (gridSize + blockSize - 1) / blockSize;
    test_spread_infection_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);

    cudaMemcpy(h_population, d_population, gridSize * sizeof(GPUPerson),
        cudaMemcpyDeviceToHost);

    printf("\nAfter Infection Spread:\n");
    printf("S: Susceptible, I: Infectious, E: Exposed\n");
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            char state = 'S';
            if (h_population[to1D(i, j, L)].Health != S) state = 'I';
            if (h_population[to1D(i, j, L)].Swap == E) state = 'E';
            printf("%c ", state);
        }
        printf("\n");
    }

    printf("\nDetailed Statistics:\n");
    int total_susceptible = 0;
    int total_exposed = 0;
    int total_infectious = 0;
    int max_exponent = 0;

    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            int idx = to1D(i, j, L);
            if (h_population[idx].Health == S) total_susceptible++;
            if (h_population[idx].Swap == E) total_exposed++;
            if (isInfectious(h_population[idx].Health)) total_infectious++;
            if (h_population[idx].Exponent > max_exponent)
                max_exponent = h_population[idx].Exponent;
        }
    }

    printf("Total Susceptible: %d\n", total_susceptible);
    printf("Total Exposed (new infections): %d\n", total_exposed);
    printf("Total Infectious: %d\n", total_infectious);
    printf("Maximum Exposure Count: %d\n", max_exponent);

    printf("\nExposure Map (number of exposures per cell):\n");
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            printf("%2d ", h_population[to1D(i, j, L)].Exponent);
        }
        printf("\n");
    }

    delete[] h_population;
    cudaFree(d_population);
    cleanupRNG();
}
int main() {
    // Initialize CUDA device first
    cudaFree(0);  // Force CUDA initialization

    printf("Initializing simulation...\n");
    cities(ROC);  // Explicitly using Rocinha

    setupGPUConstants();
    printf("GPU Constants set up\n");


    printf("City parameters set up\n");

    test_spread_infection();

    cleanupGPUConstants();
    return 0;

}