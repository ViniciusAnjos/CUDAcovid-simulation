// test_ICU_kernel.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include "gpu_define.cuh"
#include "gpu_person.cuh"
#include "gpu_utils.cuh"
#include "gpu_aleat.cuh"
#include "update_kernel.cuh"  // For counter access

#include "ICU_kernel.cuh"

void printICUState(GPUPerson* h_population, int L, const char* title) {
    printf("\n%s:\n", title);
    printf("Grid (U=ICU, R=Recovered, C=COVID Death, D=Natural Death, S=Susceptible):\n");

    int icuCount = 0;
    int recoveredCount = 0;
    int covidDeathCount = 0;

    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            int idx = to1D(i, j, L);
            char symbol = 'S';

            switch (h_population[idx].Health) {
            case ICU: symbol = 'U'; icuCount++; break;
            case Recovered: symbol = 'R'; recoveredCount++; break;
            case DeadCovid: symbol = 'C'; covidDeathCount++; break;
            case Dead: symbol = 'D'; break;
            }
            printf("%c ", symbol);
        }
        printf("\n");
    }
    printf("Total in ICU: %d, Recovered: %d, COVID Deaths: %d\n",
        icuCount, recoveredCount, covidDeathCount);
}

void test_ICU_kernel() {
    const int L = 10;  // Small grid for testing
    const int gridSize = (L + 2) * (L + 2);

    // Allocate and initialize test population
    GPUPerson* h_population = new GPUPerson[gridSize];
    GPUPerson* d_population;
    cudaMalloc(&d_population, gridSize * sizeof(GPUPerson));

    printf("Testing ICU_kernel with %dx%d grid\n", L, L);

    // Initialize all cells
    for (int i = 0; i < gridSize; i++) {
        h_population[i].Health = S;
        h_population[i].Swap = S;
        h_population[i].TimeOnState = 0;
        h_population[i].StateTime = 0;
        h_population[i].Days = 0;
        h_population[i].AgeDays = 365 * 30;
        h_population[i].AgeYears = 30;
        h_population[i].AgeDeathDays = 365 * 80;
        h_population[i].AgeDeathYears = 80;
    }

    // Set initial available ICU beds
    int initialAvailableICUBeds = 10;
    cudaMemcpyToSymbol(AvailableBedsICU, &initialAvailableICUBeds, sizeof(int));

    printf("\nInitial ICU Resources:\n");
    printf("Available ICU Beds: %d\n", initialAvailableICUBeds);

    // Set up test cases
    printf("\nSetting up test cases:\n");

    // Case 1: Young patient in ICU (better recovery chance)
    printf("Case 1: Young patient (25yo) ready for outcome at (2,2)\n");
    h_population[to1D(2, 2, L)].Health = ICU;
    h_population[to1D(2, 2, L)].Swap = ICU;
    h_population[to1D(2, 2, L)].AgeYears = 25;
    h_population[to1D(2, 2, L)].StateTime = 15;
    h_population[to1D(2, 2, L)].TimeOnState = 15; // Ready for transition

    // Case 2: Middle-aged patient mid-stay
    printf("Case 2: Middle-aged patient (50yo) mid-ICU stay at (3,3)\n");
    h_population[to1D(3, 3, L)].Health = ICU;
    h_population[to1D(3, 3, L)].Swap = ICU;
    h_population[to1D(3, 3, L)].AgeYears = 50;
    h_population[to1D(3, 3, L)].StateTime = 30;
    h_population[to1D(3, 3, L)].TimeOnState = 10; // Not ready yet

    // Case 3: Elderly patient ready for outcome (low recovery probability)
    printf("Case 3: Elderly patient (70yo) ready for outcome at (4,4)\n");
    h_population[to1D(4, 4, L)].Health = ICU;
    h_population[to1D(4, 4, L)].Swap = ICU;
    h_population[to1D(4, 4, L)].AgeYears = 70;
    h_population[to1D(4, 4, L)].StateTime = 15;
    h_population[to1D(4, 4, L)].TimeOnState = 15;

    // Case 4: Very elderly patient ready for outcome
    printf("Case 4: Very elderly patient (85yo) ready for outcome at (5,5)\n");
    h_population[to1D(5, 5, L)].Health = ICU;
    h_population[to1D(5, 5, L)].Swap = ICU;
    h_population[to1D(5, 5, L)].AgeYears = 85;
    h_population[to1D(5, 5, L)].StateTime = 15;
    h_population[to1D(5, 5, L)].TimeOnState = 15;

    // Case 5: Natural death case
    printf("Case 5: Natural death case at (6,6)\n");
    h_population[to1D(6, 6, L)].Health = ICU;
    h_population[to1D(6, 6, L)].Swap = ICU;
    h_population[to1D(6, 6, L)].AgeYears = 78;
    h_population[to1D(6, 6, L)].Days = 85 * 365; // Past death age
    h_population[to1D(6, 6, L)].AgeDeathDays = 80 * 365;

    // Case 6: Another elderly patient
    printf("Case 6: Another elderly patient (75yo) at (7,7)\n");
    h_population[to1D(7, 7, L)].Health = ICU;
    h_population[to1D(7, 7, L)].Swap = ICU;
    h_population[to1D(7, 7, L)].AgeYears = 75;
    h_population[to1D(7, 7, L)].StateTime = 15;
    h_population[to1D(7, 7, L)].TimeOnState = 15;

    // Case 7: Very old patient to test multiple outcomes
    printf("Case 7: Very old patient (90yo) at (8,8)\n");
    h_population[to1D(8, 8, L)].Health = ICU;
    h_population[to1D(8, 8, L)].Swap = ICU;
    h_population[to1D(8, 8, L)].AgeYears = 90;
    h_population[to1D(8, 8, L)].StateTime = 15;
    h_population[to1D(8, 8, L)].TimeOnState = 15;

    // Copy to device
    cudaMemcpy(d_population, h_population, gridSize * sizeof(GPUPerson),
        cudaMemcpyHostToDevice);

    // Setup RNG
    unsigned int* d_rngStates;
    cudaMalloc(&d_rngStates, gridSize * sizeof(unsigned int));
    int blockSize = 256;
    int numBlocks = (gridSize + blockSize - 1) / blockSize;
    unsigned int seed = 893221891;
    initRNG << <numBlocks, blockSize >> > (d_rngStates, seed, gridSize);
    cudaDeviceSynchronize();

    // Print initial state
    printICUState(h_population, L, "Initial State");

    // Print ICU recovery probabilities for reference
    printf("\nICU Recovery Probabilities (from original data):\n");
    printf("Age <60: ~%.3f\n", 0.5);      // Young have 50% recovery in ICU
    printf("Age 60-70: ~%.3f\n", 0.179);  // From original data
    printf("Age 70-80: ~%.3f\n", 0.125);  // From original data
    printf("Age 80-90: ~%.3f\n", 0.104);  // From original data
    printf("Age >90: ~%.3f\n", 0.083);    // From original data

    // Reset counters
    resetNewCounters_kernel << <1, 1 >> > ();
    cudaDeviceSynchronize();

    // Run ICU_kernel
    printf("\nRunning ICU_kernel...\n");
    ICU_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(h_population, d_population, gridSize * sizeof(GPUPerson),
        cudaMemcpyDeviceToHost);

    // Get final bed counts and statistics
    int finalAvailableICUBeds;
    cudaMemcpyFromSymbol(&finalAvailableICUBeds, AvailableBedsICU, sizeof(int));

    int h_new_cases[15] = { 0 };
    int dummy_totals[15] = { 0 };
    getCountersFromDevice(dummy_totals, h_new_cases);

    // Print final state (showing Swap values)
    printf("\nFinal State After ICU_kernel (showing Swap values):\n");
    printf("Grid (U=ICU, R=Recovered, C=COVID Death, D=Natural Death, S=Susceptible):\n");
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            int idx = to1D(i, j, L);
            char symbol = 'S';

            // Show Swap values to see transitions
            switch (h_population[idx].Swap) {
            case ICU: symbol = 'U'; break;
            case Recovered: symbol = 'R'; break;
            case DeadCovid: symbol = 'C'; break;
            case Dead: symbol = 'D'; break;
            }
            printf("%c ", symbol);
        }
        printf("\n");
    }

    printf("\nResource Changes:\n");
    printf("ICU Beds: %d → %d (freed: %d)\n",
        initialAvailableICUBeds, finalAvailableICUBeds,
        finalAvailableICUBeds - initialAvailableICUBeds);

    printf("\nNew Case Counts:\n");
    printf("New Recovered from ICU: %d\n", h_new_cases[Recovered]);
    printf("New COVID Deaths in ICU: %d\n", h_new_cases[DeadCovid]);

    // Print individual case results
    printf("\nIndividual Case Results:\n");
    for (int caseNum = 1; caseNum <= 7; caseNum++) {
        int i, j;
        switch (caseNum) {
        case 1: i = 2; j = 2; break;
        case 2: i = 3; j = 3; break;
        case 3: i = 4; j = 4; break;
        case 4: i = 5; j = 5; break;
        case 5: i = 6; j = 6; break;
        case 6: i = 7; j = 7; break;
        case 7: i = 8; j = 8; break;
        }

        int idx = to1D(i, j, L);
        const char* outcome = "Unknown";
        if (h_population[idx].Swap == ICU) outcome = "Still in ICU";
        else if (h_population[idx].Swap == Recovered) outcome = "Recovered";
        else if (h_population[idx].Swap == DeadCovid) outcome = "Died (COVID)";
        else if (h_population[idx].Swap == Dead) outcome = "Died (Natural)";

        printf("Case %d (%d,%d): Age %d → %s\n",
            caseNum, i, j, h_population[idx].AgeYears, outcome);
    }

    // Summary statistics
    int totalRecovered = 0;
    int totalDiedCovid = 0;
    int totalDiedNatural = 0;
    int stillInICU = 0;

    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            int idx = to1D(i, j, L);
            if (h_population[idx].Health == ICU) {
                switch (h_population[idx].Swap) {
                case Recovered: totalRecovered++; break;
                case DeadCovid: totalDiedCovid++; break;
                case Dead: totalDiedNatural++; break;
                case ICU: stillInICU++; break;
                }
            }
        }
    }

    printf("\nSummary:\n");
    printf("Total outcomes from 7 ICU patients:\n");
    printf("- Recovered: %d\n", totalRecovered);
    printf("- Died from COVID: %d\n", totalDiedCovid);
    printf("- Died naturally: %d\n", totalDiedNatural);
    printf("- Still in ICU: %d\n", stillInICU);

    // Cleanup
    delete[] h_population;
    cudaFree(d_population);
    cudaFree(d_rngStates);
}

int main() {
    // Initialize city and GPU constants
    int city = ROC;
    setupCityParameters(city);
    setupGPUConstants();

    // Run the test
    test_ICU_kernel();

    // Cleanup
    cleanupGPUConstants();

    return 0;
}