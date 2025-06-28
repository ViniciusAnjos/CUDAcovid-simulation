// covid_simulation_complete.cu
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

// Include all state kernels
#include "S_kernel.cuh"
#include "E_kernel.cuh"
#include "IP_kernel.cuh"
#include "IS_kernel.cuh"
#include "H_kernel.cuh"
#include "ICU_kernel.cuh"

// Function to run one simulation day
void runSimulationDay(GPUPerson* d_population, unsigned int* d_rngStates,
    int L, int day, int blockSize, int numBlocks) {

    printf("\n--- Day %d ---\n", day);

    // Update boundaries
    printf("Updating boundaries...\n");
    updateBoundaries_kernel << <numBlocks, blockSize >> > (d_population, L);
    cudaDeviceSynchronize();

    // Run state kernels
    printf("Running S kernel...\n");
    S_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);
    cudaDeviceSynchronize();

    printf("Running E kernel...\n");
    E_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);
    cudaDeviceSynchronize();

    printf("Running IP kernel...\n");
    IP_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);
    cudaDeviceSynchronize();

    printf("Running IS kernel...\n");
    IS_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);
    cudaDeviceSynchronize();

    printf("Running H kernel...\n");
    H_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);
    cudaDeviceSynchronize();

    printf("Running ICU kernel...\n");
    ICU_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);
    cudaDeviceSynchronize();

    // Reset counters for update kernel
    resetCounters_kernel << <1, 1 >> > ();
    resetNewCounters_kernel << <1, 1 >> > ();
    cudaDeviceSynchronize();

    // Run update kernel
    printf("Running update kernel...\n");
    update_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L, day, d_ProbNaturalDeath);
    cudaDeviceSynchronize();

    // Get statistics
    int h_totals[15] = { 0 };
    int h_new_cases[15] = { 0 };
    getCountersFromDevice(h_totals, h_new_cases);

    // Print key statistics
    printf("Day %d statistics:\n", day);
    printf("  S: %d (new: %d)\n", h_totals[S], h_new_cases[S]);
    printf("  E: %d (new: %d)\n", h_totals[E], h_new_cases[E]);
    printf("  IP: %d (new: %d)\n", h_totals[IP], h_new_cases[IP]);
    printf("  IA: %d (new: %d)\n", h_totals[IA], h_new_cases[IA]);
    printf("  IS (L/M/S): %d/%d/%d\n", h_totals[ISLight], h_totals[ISModerate], h_totals[ISSevere]);
    printf("  H: %d, ICU: %d\n", h_totals[H], h_totals[ICU]);
    printf("  Recovered: %d, Deaths: %d\n", h_totals[Recovered], h_totals[DeadCovid]);

    // Reset new counters for next day
    resetNewCounters_kernel << <1, 1 >> > ();
    cudaDeviceSynchronize();
}

// Add this function to check if probability arrays are correctly set

__global__ void checkProbabilities_kernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n=== CHECKING PROBABILITY ARRAYS ===\n");

        printf("Recovery probabilities for severe cases:\n");
        printf("Age 25: %.6f (should be ~0.01)\n", d_ProbRecoverySevere[25]);
        printf("Age 65: %.6f (should be ~0.00357)\n", d_ProbRecoverySevere[65]);
        printf("Age 75: %.6f (should be ~0.00250)\n", d_ProbRecoverySevere[75]);
        printf("Age 85: %.6f (should be ~0.00125)\n", d_ProbRecoverySevere[85]);
        printf("Age 95: %.6f (should be ~0.00167)\n", d_ProbRecoverySevere[95]);

        printf("\nNatural death probabilities:\n");
        printf("Age 30: %.6f\n", d_ProbNaturalDeath[30]);
        printf("Age 60: %.6f\n", d_ProbNaturalDeath[60]);
        printf("Age 80: %.6f\n", d_ProbNaturalDeath[80]);

        printf("\nAge structure probabilities:\n");
        printf("Age group 0 (0-4): %.6f\n", d_ProbBirthAge[1]);
        printf("Age group 17 (80-84): %.6f\n", d_ProbBirthAge[17]);
        printf("Cumulative prob[0]: %.6f\n", d_SumProbBirthAge[0]);
        printf("Cumulative prob[17]: %.6f\n", d_SumProbBirthAge[17]);
    }
}

void checkGPUProbabilities() {
    printf("Checking GPU probability arrays...\n");
    checkProbabilities_kernel << <1, 1 >> > ();
    cudaDeviceSynchronize();
}

void testDeathMechanisms() {
    printf("\n=== TESTING DEATH MECHANISMS ===\n");

    const int L = 10;
    const int gridSize = (L + 2) * (L + 2);

    GPUPerson* h_population = new GPUPerson[gridSize];
    GPUPerson* d_population;
    cudaMalloc(&d_population, gridSize * sizeof(GPUPerson));

    // Initialize with specific test cases
    for (int i = 0; i < gridSize; i++) {
        h_population[i].Health = S;
        h_population[i].Swap = S;
        h_population[i].AgeYears = 30;
        h_population[i].Days = 0;
        h_population[i].AgeDeathDays = 365 * 80;
        h_population[i].TimeOnState = 0;
        h_population[i].StateTime = 5;
    }

    // Force some ISSevere cases with elderly people
    h_population[to1D(2, 2, L)].Health = ISSevere;
    h_population[to1D(2, 2, L)].AgeYears = 75;  // Elderly
    h_population[to1D(2, 2, L)].TimeOnState = 10; // Ready to transition
    h_population[to1D(2, 2, L)].StateTime = 5;

    h_population[to1D(3, 3, L)].Health = ISSevere;
    h_population[to1D(3, 3, L)].AgeYears = 85;  // Very elderly
    h_population[to1D(3, 3, L)].TimeOnState = 10;
    h_population[to1D(3, 3, L)].StateTime = 5;

    h_population[to1D(4, 4, L)].Health = ISSevere;
    h_population[to1D(4, 4, L)].AgeYears = 90;  // Extremely elderly
    h_population[to1D(4, 4, L)].TimeOnState = 10;
    h_population[to1D(4, 4, L)].StateTime = 5;

    // Force natural death case
    h_population[to1D(5, 5, L)].Health = S;
    h_population[to1D(5, 5, L)].Days = 365 * 85;  // 85 years old in days
    h_population[to1D(5, 5, L)].AgeDeathDays = 365 * 80;  // Should die

    printf("Test setup:\n");
    printf("- Person (2,2): ISSevere, Age 75, ready to transition\n");
    printf("- Person (3,3): ISSevere, Age 85, ready to transition\n");
    printf("- Person (4,4): ISSevere, Age 90, ready to transition\n");
    printf("- Person (5,5): Natural death case\n");

    // Copy to device
    cudaMemcpy(d_population, h_population, gridSize * sizeof(GPUPerson), cudaMemcpyHostToDevice);

    // Set hospital beds to 0 to force deaths
    int zeroBeds = 0;
    cudaMemcpyToSymbol(AvailableBeds, &zeroBeds, sizeof(int));

    // Setup RNG
    unsigned int* d_rngStates;
    cudaMalloc(&d_rngStates, gridSize * sizeof(unsigned int));
    int blockSize = 256;
    int numBlocks = (gridSize + blockSize - 1) / blockSize;
    initRNG << <numBlocks, blockSize >> > (d_rngStates, 12345, gridSize);
    cudaDeviceSynchronize();

    printf("\nRunning IS_kernel with 0 hospital beds...\n");

    // Run the debug IS kernel
    IS_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);
    cudaDeviceSynchronize();

    // Copy back and check results
    cudaMemcpy(h_population, d_population, gridSize * sizeof(GPUPerson), cudaMemcpyDeviceToHost);

    printf("\nResults:\n");
    printf("Person (2,2): Health=%d, Swap=%d, Age=%d\n",
        h_population[to1D(2, 2, L)].Health, h_population[to1D(2, 2, L)].Swap, h_population[to1D(2, 2, L)].AgeYears);
    printf("Person (3,3): Health=%d, Swap=%d, Age=%d\n",
        h_population[to1D(3, 3, L)].Health, h_population[to1D(3, 3, L)].Swap, h_population[to1D(3, 3, L)].AgeYears);
    printf("Person (4,4): Health=%d, Swap=%d, Age=%d\n",
        h_population[to1D(4, 4, L)].Health, h_population[to1D(4, 4, L)].Swap, h_population[to1D(4, 4, L)].AgeYears);
    printf("Person (5,5): Health=%d, Swap=%d, Days=%d, AgeDeathDays=%d\n",
        h_population[to1D(5, 5, L)].Health, h_population[to1D(5, 5, L)].Swap,
        h_population[to1D(5, 5, L)].Days, h_population[to1D(5, 5, L)].AgeDeathDays);

    printf("Expected: Swap should be %d (DeadCovid) or %d (Dead)\n", DeadCovid, Dead);

    // Cleanup
    delete[] h_population;
    cudaFree(d_population);
    cudaFree(d_rngStates);
}

int main(int argc, char* argv[]) {
    printf("COVID-19 CUDA Simulation - Complete Version\n");

    // Initialize city and GPU constants
    int city = ROC;  // Rocinha
    setupCityParameters(city);
    setupGPUConstants();
    checkGPUProbabilities();
    testDeathMechanisms();
    // Simulation parameters
    int simulationNumber = 1;
    const int L = 100;  // Grid size
    const int gridSize = (L + 2) * (L + 2);
    const int N = L * L;
    const int DAYS_TO_RUN = 1;  // Run for 10 days as a test

    printf("Grid size: %d x %d = %d cells\n", L, L, N);
    printf("Running for %d days\n", DAYS_TO_RUN);

    // Allocate device memory
    GPUPerson* d_population;
    cudaMalloc(&d_population, gridSize * sizeof(GPUPerson));

    // Initialize RNG
    unsigned int* d_rngStates;
    cudaMalloc(&d_rngStates, gridSize * sizeof(unsigned int));

    unsigned int seed = 893221891 * simulationNumber;
    int blockSize = 256;
    int numBlocks = (gridSize + blockSize - 1) / blockSize;

    initRNG << <numBlocks, blockSize >> > (d_rngStates, seed, gridSize);
    cudaDeviceSynchronize();

    // Initialize population
    printf("\nInitializing population...\n");
    initPopulation_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);
    cudaDeviceSynchronize();

    // Initialize counters
    int* d_stateCounts, * d_newCounts;
    cudaMalloc(&d_stateCounts, 15 * sizeof(int));
    cudaMalloc(&d_newCounts, 15 * sizeof(int));

    initCounters_kernel << <1, 32 >> > (d_stateCounts, d_newCounts, N);
    cudaDeviceSynchronize();

    // Distribute initial infections
    printf("Distributing initial infections (5 IP cases)...\n");
    distributeInitialInfections_kernel << <1, 1 >> > (
        d_population, d_rngStates, d_stateCounts, d_newCounts, L,
        0,  // Eini
        5,  // IPini
        0,  // IAini
        0,  // ISLightini
        0,  // ISModerateini
        2000   // ISSevereini
        );
    cudaDeviceSynchronize();

    // Set available beds
    int availableBeds = NumberOfHospitalBeds - NumberOfHospitalBeds * AverageOcupationRateBeds;
    int availableBedsICU = NumberOfICUBeds - NumberOfICUBeds * AverageOcupationRateBedsICU;
    cudaMemcpyToSymbol(AvailableBeds, &availableBeds, sizeof(int));
    cudaMemcpyToSymbol(AvailableBedsICU, &availableBedsICU, sizeof(int));

    printf("\nStarting simulation...\n");
    printf("Available beds: Hospital=%d, ICU=%d\n", availableBeds, availableBedsICU);

    // Run simulation for specified days
    for (int day = 1; day <= DAYS_TO_RUN; day++) {
        runSimulationDay(d_population, d_rngStates, L, day, blockSize, numBlocks);
    }

    // Final statistics
    printf("\n=== Final Statistics ===\n");
    int h_totals[15] = { 0 };
    int h_new_cases[15] = { 0 };
    getCountersFromDevice(h_totals, h_new_cases);

    int totalInfectious = h_totals[ISLight] + h_totals[ISModerate] + h_totals[ISSevere];
    int totalPopulation = 0;
    for (int i = 0; i < 15; i++) {
        totalPopulation += h_totals[i];
    }

    printf("Total population check: %d (should be close to %d)\n", totalPopulation, N);
    printf("Total infectious: %d\n", totalInfectious);
    printf("Total recovered: %d\n", h_totals[Recovered]);
    printf("Total COVID deaths: %d\n", h_totals[DeadCovid]);

    // Cleanup
    printf("\nCleaning up...\n");
    cudaFree(d_population);
    cudaFree(d_rngStates);
    cudaFree(d_stateCounts);
    cudaFree(d_newCounts);
    cleanupGPUConstants();

    printf("\nSimulation completed successfully!\n");

    return 0;
}