// Corrected covid.cu - Using existing output_files.cuh system
// Fixes for Day 0 double-counting and DeadCovid initialization issues

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
#include "output_files.cuh"  // Use existing output system

// Include all state kernels
#include "S_kernel.cuh"
#include "E_kernel.cuh"
#include "IP_kernel.cuh"
#include "IS_kernel.cuh"
#include "H_kernel.cuh"
#include "ICU_kernel.cuh"

// R0 accumulator across simulations
double R0_Sum = 0.0;

// Arrays for storing simulation results across multiple simulations
double S_Sum[DAYS + 2] = { 0 };
double E_Sum[DAYS + 2] = { 0 };
double IP_Sum[DAYS + 2] = { 0 };
double IA_Sum[DAYS + 2] = { 0 };
double ISLight_Sum[DAYS + 2] = { 0 };
double ISModerate_Sum[DAYS + 2] = { 0 };
double ISSevere_Sum[DAYS + 2] = { 0 };
double H_Sum[DAYS + 2] = { 0 };
double ICU_Sum[DAYS + 2] = { 0 };
double Recovered_Sum[DAYS + 2] = { 0 };
double DeadCovid_Sum[DAYS + 2] = { 0 };

double New_S_Sum[DAYS + 2] = { 0 };
double New_E_Sum[DAYS + 2] = { 0 };
double New_IP_Sum[DAYS + 2] = { 0 };
double New_IA_Sum[DAYS + 2] = { 0 };
double New_ISLight_Sum[DAYS + 2] = { 0 };
double New_ISModerate_Sum[DAYS + 2] = { 0 };
double New_ISSevere_Sum[DAYS + 2] = { 0 };
double New_H_Sum[DAYS + 2] = { 0 };
double New_ICU_Sum[DAYS + 2] = { 0 };
double New_Recovered_Sum[DAYS + 2] = { 0 };
double New_DeadCovid_Sum[DAYS + 2] = { 0 };

// Mean arrays for final output
double S_Mean[DAYS + 2];
double E_Mean[DAYS + 2];
double IP_Mean[DAYS + 2];
double IA_Mean[DAYS + 2];
double ISLight_Mean[DAYS + 2];
double ISModerate_Mean[DAYS + 2];
double ISSevere_Mean[DAYS + 2];
double H_Mean[DAYS + 2];
double ICU_Mean[DAYS + 2];
double Recovered_Mean[DAYS + 2];
double DeadCovid_Mean[DAYS + 2];

double New_S_Mean[DAYS + 2];
double New_E_Mean[DAYS + 2];
double New_IP_Mean[DAYS + 2];
double New_IA_Mean[DAYS + 2];
double New_ISLight_Mean[DAYS + 2];
double New_ISModerate_Mean[DAYS + 2];
double New_ISSevere_Mean[DAYS + 2];
double New_H_Mean[DAYS + 2];
double New_ICU_Mean[DAYS + 2];
double New_Recovered_Mean[DAYS + 2];
double New_DeadCovid_Mean[DAYS + 2];

// Function to run one simulation day
void runSimulationDay(GPUPerson* d_population, unsigned int* d_rngStates,
    int L, int day, int blockSize, int numBlocks) {

    // Update boundaries
    updateBoundaries_kernel << <numBlocks, blockSize >> > (d_population, L);
    cudaDeviceSynchronize();

    // Run state kernels
    S_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);
    cudaDeviceSynchronize();

    E_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);
    cudaDeviceSynchronize();

    IP_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);
    cudaDeviceSynchronize();

    IS_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);
    cudaDeviceSynchronize();

    H_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);
    cudaDeviceSynchronize();

    ICU_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);
    cudaDeviceSynchronize();

    // Reset counters and run update kernel
    resetCounters_kernel << <1, 1 >> > ();
    resetNewCounters_kernel << <1, 1 >> > ();
    cudaDeviceSynchronize();

    update_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L, day, d_ProbNaturalDeath);
    cudaDeviceSynchronize();
}

int main(int argc, char* argv[]) {
    printf("COVID-19 CUDA Simulation - Complete Version with File Output\n");

    // Initialize city and GPU constants
    int city = SP;  // Rocinha
    setupCityParameters(city);
    setupGPUConstants();

    // Simulation parameters
    const int L = 3355;  // S�o Paulo (L do monografia)
    const int gridSize = (L + 2) * (L + 2);
    const int N = L * L;
    const int DAYS_TO_RUN = 200;  // limite seguro: pior caso ~150 dias
    const int MAXSIM = 1000;

    printf("Grid size: %d x %d = %d cells\n", L, L, N);
    printf("Running for %d days, %d simulations\n", DAYS_TO_RUN, MAXSIM);

    // Initialize output files using existing system
    initializeOutputFiles();

    // Allocate device memory
    GPUPerson* d_population;
    cudaMalloc(&d_population, gridSize * sizeof(GPUPerson));

    // Initialize RNG
    unsigned int* d_rngStates;
    cudaMalloc(&d_rngStates, gridSize * sizeof(unsigned int));

    int blockSize = 256;
    int numBlocks = (gridSize + blockSize - 1) / blockSize;

    // Inicializa populacao UMA VEZ (idades, estrutura demografica)
    initRNG<<<numBlocks, blockSize>>>(d_rngStates, 893221891u, gridSize);
    cudaDeviceSynchronize();
    initPopulation_kernel<<<numBlocks, blockSize>>>(d_population, d_rngStates, L);
    cudaDeviceSynchronize();
    printf("Populacao inicializada. Iniciando %d simulacoes (Beta=%.4f)...\n",
           MAXSIM, Beta);

    int h_totals[15] = { 0 };
    int h_new_cases[15] = { 0 };

    // Loop de simulacoes para calibracao R0
    for (int simulation = 1; simulation <= MAXSIM; simulation++) {

        // Reset barato: apenas estados de saude, sem recriar idades
        resetForR0_kernel<<<numBlocks, blockSize>>>(d_population, d_rngStates, L);
        cudaDeviceSynchronize();

        // RNG com seed unico por simulacao
        initRNG<<<numBlocks, blockSize>>>(d_rngStates, 893221891u * simulation, gridSize);
        cudaDeviceSynchronize();

        // Coloca 1 paciente zero (IP) e reseta contadores/flags
        placePatientZero_kernel<<<1, 1>>>(d_population, d_rngStates, L);
        initSimulationCounters_kernel<<<1, 1>>>(N);
        cudaDeviceSynchronize();

        // Loop diario — para quando paciente zero se recupera/morre
        for (int day = 1; day <= DAYS_TO_RUN; day++) {
            resetCounters_kernel<<<1, 1>>>();
            resetNewCounters_kernel<<<1, 1>>>();
            cudaDeviceSynchronize();

            runSimulationDay(d_population, d_rngStates, L, day, blockSize, numBlocks);

#ifdef PATIENT_ZERO_ONLY_MODE
            if (getPatientZeroActiveFromDevice() == 0) break;
#endif
        }

        // Acumula R0 desta simulacao
        R0_Sum += (double)getR0CountFromDevice();
    }

    // R0 medio
    double R0_Mean = R0_Sum / (double)MAXSIM;
    printf("\n=== R0 medio (Beta=%.4f): %.2f ===\n", Beta, R0_Mean);
    printf("Alvo: R0 = 3.5\n");

    // Calculate means across all simulations
    for (int t = 1; t <= DAYS_TO_RUN; t++) {
        S_Mean[t] = S_Sum[t] / (double)MAXSIM;
        E_Mean[t] = E_Sum[t] / (double)MAXSIM;
        IP_Mean[t] = IP_Sum[t] / (double)MAXSIM;
        IA_Mean[t] = IA_Sum[t] / (double)MAXSIM;
        ISLight_Mean[t] = ISLight_Sum[t] / (double)MAXSIM;
        ISModerate_Mean[t] = ISModerate_Sum[t] / (double)MAXSIM;
        ISSevere_Mean[t] = ISSevere_Sum[t] / (double)MAXSIM;
        H_Mean[t] = H_Sum[t] / (double)MAXSIM;
        ICU_Mean[t] = ICU_Sum[t] / (double)MAXSIM;
        Recovered_Mean[t] = Recovered_Sum[t] / (double)MAXSIM;
        DeadCovid_Mean[t] = DeadCovid_Sum[t] / (double)MAXSIM;

        New_S_Mean[t] = New_S_Sum[t] / (double)MAXSIM;
        New_E_Mean[t] = New_E_Sum[t] / (double)MAXSIM;
        New_IP_Mean[t] = New_IP_Sum[t] / (double)MAXSIM;
        New_IA_Mean[t] = New_IA_Sum[t] / (double)MAXSIM;
        New_ISLight_Mean[t] = New_ISLight_Sum[t] / (double)MAXSIM;
        New_ISModerate_Mean[t] = New_ISModerate_Sum[t] / (double)MAXSIM;
        New_ISSevere_Mean[t] = New_ISSevere_Sum[t] / (double)MAXSIM;
        New_H_Mean[t] = New_H_Sum[t] / (double)MAXSIM;
        New_ICU_Mean[t] = New_ICU_Sum[t] / (double)MAXSIM;
        New_Recovered_Mean[t] = New_Recovered_Sum[t] / (double)MAXSIM;
        New_DeadCovid_Mean[t] = New_DeadCovid_Sum[t] / (double)MAXSIM;
    }

    // Write final averaged results using existing output system
    writeFinalAveragedResults(S_Mean, E_Mean, IP_Mean, IA_Mean, ISLight_Mean,
        ISModerate_Mean, ISSevere_Mean, H_Mean, ICU_Mean,
        Recovered_Mean, DeadCovid_Mean,
        New_S_Mean, New_E_Mean, New_IP_Mean, New_IA_Mean,
        New_ISLight_Mean, New_ISModerate_Mean, New_ISSevere_Mean,
        New_H_Mean, New_ICU_Mean, New_Recovered_Mean,
        New_DeadCovid_Mean, DAYS_TO_RUN);

    // Close all output files using existing system
    closeOutputFiles();

    // Final statistics
    printf("\n=== Final Statistics ===\n");
    double totalInfectious = ISLight_Mean[DAYS_TO_RUN] + ISModerate_Mean[DAYS_TO_RUN] + ISSevere_Mean[DAYS_TO_RUN];

    printf("Infectious: %.4f\n", totalInfectious);


    // Cleanup
    printf("\nCleaning up...\n");
    cudaFree(d_population);
    cudaFree(d_rngStates);
    cleanupGPUConstants();

    printf("\nSimulation completed successfully!\n");
    printf("Output files are ready for analysis.\n");

    return 0;
}
