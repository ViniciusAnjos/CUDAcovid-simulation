// test_H_kernel.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include "gpu_define.cuh"
#include "gpu_person.cuh"
#include "gpu_utils.cuh"
#include "gpu_aleat.cuh"
#include "update_kernel.cuh"  // For counter access

#include "H_kernel.cuh"

void printHospitalState(GPUPerson* h_population, int L, const char* title) {
    printf("\n%s:\n", title);
    printf("Grid (H=Hospital, I=ICU, R=Recovered, D=Dead, S=Susceptible):\n");

    int hospitalCount = 0;
    int icuCount = 0;

    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            int idx = to1D(i, j, L);
            char symbol = 'S';

            switch (h_population[idx].Health) {
            case H: symbol = 'H'; hospitalCount++; break;
            case ICU: symbol = 'I'; icuCount++; break;
            case Recovered: symbol = 'R'; break;
            case Dead: symbol = 'D'; break;
            case DeadCovid: symbol = 'C'; break;
            }
            printf("%c ", symbol);
        }
        printf("\n");
    }
    printf("Total in Hospital: %d, Total in ICU: %d\n", hospitalCount, icuCount);
}

void test_H_kernel() {
    const int L = 10;  // Small grid for testing
    const int gridSize = (L + 2) * (L + 2);

    // Allocate and initialize test population
    GPUPerson* h_population = new GPUPerson[gridSize];
    GPUPerson* d_population;
    cudaMalloc(&d_population, gridSize * sizeof(GPUPerson));

    printf("Testing H_kernel (Hospital) with %dx%d grid\n", L, L);

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

    // Set initial available beds
    int initialAvailableBeds = 5;
    int initialAvailableICUBeds = 2;
    cudaMemcpyToSymbol(AvailableBeds, &initialAvailableBeds, sizeof(int));
    cudaMemcpyToSymbol(AvailableBedsICU, &initialAvailableICUBeds, sizeof(int));

    printf("\nInitial Hospital Resources:\n");
    printf("Available Hospital Beds: %d\n", initialAvailableBeds);
    printf("Available ICU Beds: %d\n", initialAvailableICUBeds);

    // Set up test cases
    printf("\nSetting up test cases:\n");

    // Case 1: Young patient ready to recover (high recovery probability)
    printf("Case 1: Young patient (25yo) ready for outcome at (2,2)\n");
    h_population[to1D(2, 2, L)].Health = H;
    h_population[to1D(2, 2, L)].Swap = H;
    h_population[to1D(2, 2, L)].AgeYears = 25;
    h_population[to1D(2, 2, L)].StateTime = 10;
    h_population[to1D(2, 2, L)].TimeOnState = 10; // Ready for transition

    // Case 2: Middle-aged patient mid-stay
    printf("Case 2: Middle-aged patient (50yo) mid-stay at (3,3)\n");
    h_population[to1D(3, 3, L)].Health = H;
    h_population[to1D(3, 3, L)].Swap = H;
    h_population[to1D(3, 3, L)].AgeYears = 50;
    h_population[to1D(3, 3, L)].StateTime = 20;
    h_population[to1D(3, 3, L)].TimeOnState = 5; // Not ready yet

    // Case 3: Elderly patient ready for outcome (lower recovery probability)
    printf("Case 3: Elderly patient (75yo) ready for outcome at (4,4)\n");
    h_population[to1D(4, 4, L)].Health = H;
    h_population[to1D(4, 4, L)].Swap = H;
    h_population[to1D(4, 4, L)].AgeYears = 75;
    h_population[to1D(4, 4, L)].StateTime = 10;
    h_population[to1D(4, 4, L)].TimeOnState = 10;

    // Case 4: Very elderly patient ready for outcome
    printf("Case 4: Very elderly patient (85yo) ready for outcome at (5,5)\n");
    h_population[to1D(5, 5, L)].Health = H;
    h_population[to1D(5, 5, L)].Swap = H;
    h_population[to1D(5, 5, L)].AgeYears = 85;
    h_population[to1D(5, 5, L)].StateTime = 10;
    h_population[to1D(5, 5, L)].TimeOnState = 10;

    // Case 5: Natural death case
    printf("Case 5: Natural death case at (6,6)\n");
    h_population[to1D(6, 6, L)].Health = H;
    h_population[to1D(6, 6, L)].Swap = H;
    h_population[to1D(6, 6, L)].AgeYears = 70;
    h_population[to1D(6, 6, L)].Days = 85 * 365; // Past death age
    h_population[to1D(6, 6, L)].AgeDeathDays = 80 * 365;

    // Case 6: Another elderly to test ICU bed shortage
    printf("Case 6: Another elderly patient (80yo) at (7,7)\n");
    h_population[to1D(7, 7, L)].Health = H;
    h_population[to1D(7, 7, L)].Swap = H;
    h_population[to1D(7, 7, L)].AgeYears = 80;
    h_population[to1D(7, 7, L)].StateTime = 10;
    h_population[to1D(7, 7, L)].TimeOnState = 10;

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
    printHospitalState(h_population, L, "Initial State");

    // Print recovery probabilities for reference
    printf("\nRecovery Probabilities:\n");
    printf("Age 25: ~%.3f\n", 1.0);  // Young have very high recovery
    printf("Age 50: ~%.3f\n", 0.897); // From original data
    printf("Age 75: ~%.3f\n", 0.678); // From original data
    printf("Age 85: ~%.3f\n", 0.457); // From original data

    // Reset counters
    resetNewCounters_kernel << <1, 1 >> > ();
    cudaDeviceSynchronize();

    // Run H_kernel
    printf("\nRunning H_kernel...\n");
    H_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);

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
    int finalAvailableBeds, finalAvailableICUBeds;
    cudaMemcpyFromSymbol(&finalAvailableBeds, AvailableBeds, sizeof(int));
    cudaMemcpyFromSymbol(&finalAvailableICUBeds, AvailableBedsICU, sizeof(int));

    int h_new_cases[15] = { 0 };
    int dummy_totals[15] = { 0 };
    getCountersFromDevice(dummy_totals, h_new_cases);

    // Print final state
    printHospitalState(h_population, L, "Final State After H_kernel");

    printf("\nResource Changes:\n");
    printf("Hospital Beds: %d → %d (change: %d)\n",
        initialAvailableBeds, finalAvailableBeds,
        finalAvailableBeds - initialAvailableBeds);
    printf("ICU Beds: %d → %d (change: %d)\n",
        initialAvailableICUBeds, finalAvailableICUBeds,
        finalAvailableICUBeds - initialAvailableICUBeds);

    printf("\nNew Case Counts:\n");
    printf("New Recovered: %d\n", h_new_cases[Recovered]);
    printf("New ICU: %d\n", h_new_cases[ICU]);
    printf("New COVID Deaths: %d\n", h_new_cases[DeadCovid]);

    // Print individual case results
    printf("\nIndividual Case Results:\n");
    for (int caseNum = 1; caseNum <= 6; caseNum++) {
        int i, j;
        switch (caseNum) {
        case 1: i = 2; j = 2; break;
        case 2: i = 3; j = 3; break;
        case 3: i = 4; j = 4; break;
        case 4: i = 5; j = 5; break;
        case 5: i = 6; j = 6; break;
        case 6: i = 7; j = 7; break;
        }

        int idx = to1D(i, j, L);
        const char* outcome = "Unknown";
        if (h_population[idx].Swap == H) outcome = "Still in Hospital";
        else if (h_population[idx].Swap == Recovered) outcome = "Recovered";
        else if (h_population[idx].Swap == ICU) outcome = "Moved to ICU";
        else if (h_population[idx].Swap == DeadCovid) outcome = "Died (COVID)";
        else if (h_population[idx].Swap == Dead) outcome = "Died (Natural)";

        printf("Case %d (%d,%d): Age %d → %s\n",
            caseNum, i, j, h_population[idx].AgeYears, outcome);
    }

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
    test_H_kernel();

    // Cleanup
    cleanupGPUConstants();

    return 0;
}