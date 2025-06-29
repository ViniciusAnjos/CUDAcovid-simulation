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
    const int L = 3200;  // Grid size
    const int gridSize = (L + 2) * (L + 2);
    const int N = L * L;
    const int DAYS_TO_RUN = 400;   // Change to 200 for full simulation
    const int MAXSIM = 10;         // Change to 5 for full averaging

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

    // CORRECTED: Initialize sum arrays to zero
    for (int t = 0; t <= DAYS_TO_RUN; t++) {
        S_Sum[t] = E_Sum[t] = IP_Sum[t] = IA_Sum[t] = 0.0;
        ISLight_Sum[t] = ISModerate_Sum[t] = ISSevere_Sum[t] = 0.0;
        H_Sum[t] = ICU_Sum[t] = Recovered_Sum[t] = DeadCovid_Sum[t] = 0.0;

        New_S_Sum[t] = New_E_Sum[t] = New_IP_Sum[t] = New_IA_Sum[t] = 0.0;
        New_ISLight_Sum[t] = New_ISModerate_Sum[t] = New_ISSevere_Sum[t] = 0.0;
        New_H_Sum[t] = New_ICU_Sum[t] = New_Recovered_Sum[t] = New_DeadCovid_Sum[t] = 0.0;
    }

    // Run multiple simulations for averaging
    for (int simulation = 1; simulation <= MAXSIM; simulation++) {
        printf("\n=== Simulation %d/%d ===\n", simulation, MAXSIM);

        // CORRECTED: Reset all counters before each simulation
        resetCounters_kernel << <1, 1 >> > ();
        resetNewCounters_kernel << <1, 1 >> > ();
        cudaDeviceSynchronize();

        // Initialize RNG with unique seed per simulation
        unsigned int seed = 893221891 * simulation;
        initRNG << <numBlocks, blockSize >> > (d_rngStates, seed, gridSize);
        cudaDeviceSynchronize();

        // Initialize population
        initPopulation_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);
        cudaDeviceSynchronize();

        // Initialize counters
        int* d_stateCounts, * d_newCounts;
        cudaMalloc(&d_stateCounts, 15 * sizeof(int));
        cudaMalloc(&d_newCounts, 15 * sizeof(int));
        initCounters_kernel << <1, 32 >> > (d_stateCounts, d_newCounts, N);
        cudaDeviceSynchronize();

        // Distribute initial infections
        distributeInitialInfections_kernel << <1, 1 >> > (
            d_population, d_rngStates, d_stateCounts, d_newCounts, L,
            0,  // Eini
            5,  // IPini
            0,  // IAini
            0,  // ISLightini
            0,  // ISModerateini
            0   // ISSevereini
            );
        cudaDeviceSynchronize();

        // Set available beds
        int availableBeds = NumberOfHospitalBeds - NumberOfHospitalBeds * AverageOcupationRateBeds;
        int availableBedsICU = NumberOfICUBeds - NumberOfICUBeds * AverageOcupationRateBedsICU;
        cudaMemcpyToSymbol(AvailableBeds, &availableBeds, sizeof(int));
        cudaMemcpyToSymbol(AvailableBedsICU, &availableBedsICU, sizeof(int));

        // CORRECTED: Get Day 0 statistics ONCE and write to files using existing system
        int h_totals[15] = { 0 };
        int h_new_cases[15] = { 0 };
        getCountersFromDevice(h_totals, h_new_cases);
        writeInitialSimulationData(simulation, h_totals, h_new_cases, N);

        // CORRECTED: Add to sum arrays for averaging (Day 0 ONLY ONCE)
        S_Sum[0] += (double)h_totals[S] / (double)N;
        E_Sum[0] += (double)h_totals[E] / (double)N;
        IP_Sum[0] += (double)h_totals[IP] / (double)N;
        IA_Sum[0] += (double)h_totals[IA] / (double)N;
        ISLight_Sum[0] += (double)h_totals[ISLight] / (double)N;
        ISModerate_Sum[0] += (double)h_totals[ISModerate] / (double)N;
        ISSevere_Sum[0] += (double)h_totals[ISSevere] / (double)N;
        H_Sum[0] += (double)h_totals[H] / (double)N;
        ICU_Sum[0] += (double)h_totals[ICU] / (double)N;
        Recovered_Sum[0] += (double)h_totals[Recovered] / (double)N;
        DeadCovid_Sum[0] += (double)h_totals[DeadCovid] / (double)N;

        New_S_Sum[0] += (double)h_new_cases[S] / (double)N;
        New_E_Sum[0] += (double)h_new_cases[E] / (double)N;
        New_IP_Sum[0] += (double)h_new_cases[IP] / (double)N;
        New_IA_Sum[0] += (double)h_new_cases[IA] / (double)N;
        New_ISLight_Sum[0] += (double)h_new_cases[ISLight] / (double)N;
        New_ISModerate_Sum[0] += (double)h_new_cases[ISModerate] / (double)N;
        New_ISSevere_Sum[0] += (double)h_new_cases[ISSevere] / (double)N;
        New_H_Sum[0] += (double)h_new_cases[H] / (double)N;
        New_ICU_Sum[0] += (double)h_new_cases[ICU] / (double)N;
        New_Recovered_Sum[0] += (double)h_new_cases[Recovered] / (double)N;
        New_DeadCovid_Sum[0] += (double)h_new_cases[DeadCovid] / (double)N;

        // CORRECTED: Run simulation starting from Day 1 (not Day 0 to avoid double-counting)
        for (int day = 1; day <= DAYS_TO_RUN; day++) {
            runSimulationDay(d_population, d_rngStates, L, day, blockSize, numBlocks);

            // Get statistics
            getCountersFromDevice(h_totals, h_new_cases);
            writeDailySimulationData(simulation, day, h_totals, h_new_cases, N);

            // Add to sum arrays for averaging
            S_Sum[day] += (double)h_totals[S] / (double)N;
            E_Sum[day] += (double)h_totals[E] / (double)N;
            IP_Sum[day] += (double)h_totals[IP] / (double)N;
            IA_Sum[day] += (double)h_totals[IA] / (double)N;
            ISLight_Sum[day] += (double)h_totals[ISLight] / (double)N;
            ISModerate_Sum[day] += (double)h_totals[ISModerate] / (double)N;
            ISSevere_Sum[day] += (double)h_totals[ISSevere] / (double)N;
            H_Sum[day] += (double)h_totals[H] / (double)N;
            ICU_Sum[day] += (double)h_totals[ICU] / (double)N;
            Recovered_Sum[day] += (double)h_totals[Recovered] / (double)N;
            DeadCovid_Sum[day] += (double)h_totals[DeadCovid] / (double)N;

            New_S_Sum[day] += (double)h_new_cases[S] / (double)N;
            New_E_Sum[day] += (double)h_new_cases[E] / (double)N;
            New_IP_Sum[day] += (double)h_new_cases[IP] / (double)N;
            New_IA_Sum[day] += (double)h_new_cases[IA] / (double)N;
            New_ISLight_Sum[day] += (double)h_new_cases[ISLight] / (double)N;
            New_ISModerate_Sum[day] += (double)h_new_cases[ISModerate] / (double)N;
            New_ISSevere_Sum[day] += (double)h_new_cases[ISSevere] / (double)N;
            New_H_Sum[day] += (double)h_new_cases[H] / (double)N;
            New_ICU_Sum[day] += (double)h_new_cases[ICU] / (double)N;
            New_Recovered_Sum[day] += (double)h_new_cases[Recovered] / (double)N;
            New_DeadCovid_Sum[day] += (double)h_new_cases[DeadCovid] / (double)N;
        }

        // Cleanup simulation-specific memory
        cudaFree(d_stateCounts);
        cudaFree(d_newCounts);
    }

    // Calculate means across all simulations
    for (int t = 0; t <= DAYS_TO_RUN; t++) {
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
    printf("Final day statistics (averaged across %d simulations):\n", MAXSIM);
    printf("Susceptible: %.4f\n", S_Mean[DAYS_TO_RUN]);
    printf("Exposed: %.4f\n", E_Mean[DAYS_TO_RUN]);
    printf("Infectious: %.4f\n", totalInfectious);
    printf("Hospitalized: %.4f\n", H_Mean[DAYS_TO_RUN]);
    printf("ICU: %.4f\n", ICU_Mean[DAYS_TO_RUN]);
    printf("Recovered: %.4f\n", Recovered_Mean[DAYS_TO_RUN]);
    printf("COVID Deaths: %.4f\n", DeadCovid_Mean[DAYS_TO_RUN]);

    // Cleanup
    printf("\nCleaning up...\n");
    cudaFree(d_population);
    cudaFree(d_rngStates);
    cleanupGPUConstants();

    printf("\nSimulation completed successfully!\n");
    printf("Output files are ready for analysis.\n");

    return 0;
}