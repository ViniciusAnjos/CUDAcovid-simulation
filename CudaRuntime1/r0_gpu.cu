// Medidor de R0 para a GPU (full-sim), mesma definicao do r0_serial.cu:
// 1 paciente-zero, conta secundarios (New_E) enquanto ele espalha (IP/IA/ISLight),
// early-stop quando IP+IA+ISLight != 1. R0 e independente de L -> usar L pequeno.
// Requer define.h: IPini=1. Cidade definida em `int city = ...` (trocada por sed).
//
//   nvcc r0_gpu.cu -o r0_gpu.exe -arch=sm_89 --diag-suppress 20091 --diag-suppress 177

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "gpu_define.cuh"
#include "gpu_person.cuh"
#include "gpu_utils.cuh"
#include "gpu_aleat.cuh"
#include "gpu_begin.cuh"
#include "gpu_neighbors.cuh"
#include "update_kernel.cuh"
#include "gpu_update_boundaries.cuh"
#include "output_files.cuh"
#include "S_kernel.cuh"
#include "E_kernel.cuh"
#include "IP_kernel.cuh"
#include "IS_kernel.cuh"
#include "H_kernel.cuh"
#include "ICU_kernel.cuh"

// Arrays globais exigidos por includes (nao usados na medicao de R0)
double S_Sum[DAYS + 2] = { 0 }, E_Sum[DAYS + 2] = { 0 }, IP_Sum[DAYS + 2] = { 0 }, IA_Sum[DAYS + 2] = { 0 };
double ISLight_Sum[DAYS + 2] = { 0 }, ISModerate_Sum[DAYS + 2] = { 0 }, ISSevere_Sum[DAYS + 2] = { 0 };
double H_Sum[DAYS + 2] = { 0 }, ICU_Sum[DAYS + 2] = { 0 }, Recovered_Sum[DAYS + 2] = { 0 }, DeadCovid_Sum[DAYS + 2] = { 0 };
double New_S_Sum[DAYS + 2] = { 0 }, New_E_Sum[DAYS + 2] = { 0 }, New_IP_Sum[DAYS + 2] = { 0 }, New_IA_Sum[DAYS + 2] = { 0 };
double New_ISLight_Sum[DAYS + 2] = { 0 }, New_ISModerate_Sum[DAYS + 2] = { 0 }, New_ISSevere_Sum[DAYS + 2] = { 0 };
double New_H_Sum[DAYS + 2] = { 0 }, New_ICU_Sum[DAYS + 2] = { 0 }, New_Recovered_Sum[DAYS + 2] = { 0 }, New_DeadCovid_Sum[DAYS + 2] = { 0 };
double S_Mean[DAYS + 2], E_Mean[DAYS + 2], IP_Mean[DAYS + 2], IA_Mean[DAYS + 2], ISLight_Mean[DAYS + 2];
double ISModerate_Mean[DAYS + 2], ISSevere_Mean[DAYS + 2], H_Mean[DAYS + 2], ICU_Mean[DAYS + 2];
double Recovered_Mean[DAYS + 2], DeadCovid_Mean[DAYS + 2];
double New_S_Mean[DAYS + 2], New_E_Mean[DAYS + 2], New_IP_Mean[DAYS + 2], New_IA_Mean[DAYS + 2], New_ISLight_Mean[DAYS + 2];
double New_ISModerate_Mean[DAYS + 2], New_ISSevere_Mean[DAYS + 2], New_H_Mean[DAYS + 2], New_ICU_Mean[DAYS + 2];
double New_Recovered_Mean[DAYS + 2], New_DeadCovid_Mean[DAYS + 2];

// roda 1 dia (igual ao covid.cu: boundaries, kernels de estado, reset, update)
void runSimulationDay(GPUPerson* d_population, unsigned int* d_rngStates,
    int L, int day, int blockSize, int numBlocks) {
    updateBoundaries_kernel << <numBlocks, blockSize >> > (d_population, L);
    cudaDeviceSynchronize();
    S_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);  cudaDeviceSynchronize();
    E_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);  cudaDeviceSynchronize();
    IP_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L); cudaDeviceSynchronize();
    IS_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L); cudaDeviceSynchronize();
    H_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);  cudaDeviceSynchronize();
    ICU_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);cudaDeviceSynchronize();
    resetCounters_kernel << <1, 1 >> > ();
    resetNewCounters_kernel << <1, 1 >> > ();
    cudaDeviceSynchronize();
    update_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L, day, d_ProbNaturalDeath);
    cudaDeviceSynchronize();
}

int main(int argc, char* argv[]) {
    int city = SP;   // <== trocado por cidade
    setupCityParameters(city);
    setupGPUConstants();

    const int gridSize = (L + 2) * (L + 2);
    const int DAYS_TO_RUN = 400;
    int blockSize = 256;
    int numBlocks = (gridSize + blockSize - 1) / blockSize;

    GPUPerson* d_population; cudaMalloc(&d_population, gridSize * sizeof(GPUPerson));
    unsigned int* d_rngStates; cudaMalloc(&d_rngStates, gridSize * sizeof(unsigned int));

    int h_totals[15] = { 0 }, h_new[15] = { 0 };
    double R0_Sum = 0.0; int zeros = 0;

    for (int sim = 1; sim <= MAXSIM; sim++) {
        initSimulationCounters_kernel << <1, 1 >> > (N); cudaDeviceSynchronize();
        unsigned int seed = 893221891u * sim;
        initRNG << <numBlocks, blockSize >> > (d_rngStates, seed, gridSize); cudaDeviceSynchronize();
        initPopulation_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L); cudaDeviceSynchronize();
        int* d_sc; int* d_nc; cudaMalloc(&d_sc, 15 * sizeof(int)); cudaMalloc(&d_nc, 15 * sizeof(int));
        initCounters_kernel << <1, 32 >> > (d_sc, d_nc, N); cudaDeviceSynchronize();
        distributeInitialInfections_kernel << <1, 1 >> > (d_population, d_rngStates, d_sc, d_nc, L,
            Eini, IPini, IAini, ISLightini, ISModerateini, ISSevereini);
        cudaDeviceSynchronize();
        int availBeds = NumberOfHospitalBeds - NumberOfHospitalBeds * AverageOcupationRateBeds;
        int availICU = NumberOfICUBeds - NumberOfICUBeds * AverageOcupationRateBedsICU;
        cudaMemcpyToSymbol(AvailableBeds, &availBeds, sizeof(int));
        cudaMemcpyToSymbol(AvailableBedsICU, &availICU, sizeof(int));

        long r0count = 0;
        int spreading = 1;   // colocamos exatamente 1 IP (IPini=1)
        for (int day = 1; day <= DAYS_TO_RUN; day++) {
            if (spreading != 1) break;   // paciente-zero saiu de IP/IA/ISLight ou secundario passou a espalhar
            runSimulationDay(d_population, d_rngStates, L, day, blockSize, numBlocks);
            getCountersFromDevice(h_totals, h_new);
            r0count += h_new[E];
            spreading = h_totals[IP] + h_totals[IA] + h_totals[ISLight];
        }
        R0_Sum += (double)r0count;
        if (r0count == 0) zeros++;
        cudaFree(d_sc); cudaFree(d_nc);
    }

    printf("=== R0 GPU (full-sim) city=%d ===\n", city);
    printf("L=%d MAXSIM=%d contatos[%.1f,%.1f] Density=%d Beta=%.5f\n",
        L, MAXSIM, MinRandomContacts, MaxRandomContacts, Density, Beta);
    printf("R0 medio = %.4f (alvo 3.5; zeros=%.1f%%)\n", R0_Sum / (double)MAXSIM, 100.0 * zeros / (double)MAXSIM);
    cudaFree(d_population); cudaFree(d_rngStates);
    return 0;
}
