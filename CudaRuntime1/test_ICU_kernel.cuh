#include <stdio.h>
#include <cuda_runtime.h>
#include "gpu_define.cuh"
#include "gpu_person.cuh"
#include "gpu_utils.cuh"
#include "gpu_aleat.cuh"

// Include the updated ICU kernel
#include "ICU_kernel.cuh"



void test_ICU_kernel_comprehensive() {
    const int L = 8;  // Larger grid for multiple test cases
    const int gridSize = (L + 2) * (L + 2);

    printf("=== Comprehensive ICU Kernel Test ===\n");
    printf("Grid size: %dx%d = %d total cells\n", L, L, L * L);

    // Setup GPU constants
    int city = ROC;  // Use Rocinha parameters
    setupCityParameters(city);
    setupGPUConstants();

    // Allocate memory
    GPUPerson* h_population = new GPUPerson[gridSize];
    GPUPerson* d_population;
    checkCudaError(cudaMalloc(&d_population, gridSize * sizeof(GPUPerson)), "Population allocation");

    // Bed counters
    int* h_hospitalBeds = new int(10);  // Start with 10 hospital beds
    int* h_icuBeds = new int(8);        // Start with 8 ICU beds
    int* d_hospitalBeds;
    int* d_icuBeds;
    checkCudaError(cudaMalloc(&d_hospitalBeds, sizeof(int)), "Hospital beds allocation");
    checkCudaError(cudaMalloc(&d_icuBeds, sizeof(int)), "ICU beds allocation");

    checkCudaError(cudaMemcpy(d_hospitalBeds, h_hospitalBeds, sizeof(int), cudaMemcpyHostToDevice), "Hospital beds copy");
    checkCudaError(cudaMemcpy(d_icuBeds, h_icuBeds, sizeof(int), cudaMemcpyHostToDevice), "ICU beds copy");

    // Initialize population
    printf("\nInitializing population...\n");
    for (int i = 0; i < gridSize; i++) {
        h_population[i].Health = S;
        h_population[i].Swap = S;
        h_population[i].TimeOnState = 0;
        h_population[i].StateTime = 0;
        h_population[i].Days = 0;
        h_population[i].AgeYears = 40;  // Default age
        h_population[i].AgeDeathDays = 365 * 80;  // Die at 80
    }

    // Set up comprehensive test cases
    printf("\nSetting up test cases:\n");

    // Case 1: Young ICU patient (high recovery chance)
    printf("Case 1: Young ICU patient (25yo) ready to transition at (2,2)\n");
    int idx = to1D(2, 2, L);
    h_population[idx].Health = ICU;
    h_population[idx].Swap = ICU;
    h_population[idx].AgeYears = 25;
    h_population[idx].StateTime = 10;
    h_population[idx].TimeOnState = 10;  // Ready to transition
    h_population[idx].Days = 25 * 365;   // 25 years old

    // Case 2: Middle-aged ICU patient (moderate recovery chance)
    printf("Case 2: Middle-aged ICU patient (55yo) ready to transition at (3,3)\n");
    idx = to1D(3, 3, L);
    h_population[idx].Health = ICU;
    h_population[idx].Swap = ICU;
    h_population[idx].AgeYears = 55;
    h_population[idx].StateTime = 15;
    h_population[idx].TimeOnState = 15;  // Ready to transition
    h_population[idx].Days = 55 * 365;

    // Case 3: Elderly ICU patient (low recovery chance)
    printf("Case 3: Elderly ICU patient (75yo) ready to transition at (4,4)\n");
    idx = to1D(4, 4, L);
    h_population[idx].Health = ICU;
    h_population[idx].Swap = ICU;
    h_population[idx].AgeYears = 75;
    h_population[idx].StateTime = 12;
    h_population[idx].TimeOnState = 12;  // Ready to transition
    h_population[idx].Days = 75 * 365;

    // Case 4: Very elderly ICU patient (very low recovery chance)
    printf("Case 4: Very elderly ICU patient (85yo) ready to transition at (5,5)\n");
    idx = to1D(5, 5, L);
    h_population[idx].Health = ICU;
    h_population[idx].Swap = ICU;
    h_population[idx].AgeYears = 85;
    h_population[idx].StateTime = 8;
    h_population[idx].TimeOnState = 8;   // Ready to transition
    h_population[idx].Days = 85 * 365;

    // Case 5: ICU patient in middle of treatment
    printf("Case 5: ICU patient (60yo) in middle of treatment at (6,6)\n");
    idx = to1D(6, 6, L);
    h_population[idx].Health = ICU;
    h_population[idx].Swap = ICU;
    h_population[idx].AgeYears = 60;
    h_population[idx].StateTime = 20;
    h_population[idx].TimeOnState = 5;   // Not ready to transition
    h_population[idx].Days = 60 * 365;

    // Case 6: Natural death case
    printf("Case 6: ICU patient (70yo) reaching natural death at (7,7)\n");
    idx = to1D(7, 7, L);
    h_population[idx].Health = ICU;
    h_population[idx].Swap = ICU;
    h_population[idx].AgeYears = 70;
    h_population[idx].StateTime = 10;
    h_population[idx].TimeOnState = 5;
    h_population[idx].Days = 85 * 365;     // 85 years in days
    h_population[idx].AgeDeathDays = 80 * 365;  // Should die at 80

    // Print recovery probabilities for reference
    printf("\nExpected Recovery Probabilities:\n");
    printf("Age 25: ~50%% (ProbRecoveryICUYounger)\n");
    printf("Age 55: ~50%% (ProbRecoveryICUYounger)\n");
    printf("Age 75: ~12.5%% (ProbRecoveryICU_70_80)\n");
    printf("Age 85: ~8.3%% (ProbRecoveryICU_Greater90)\n");

    // Print initial grid state
    printf("\nInitial Grid State (I=ICU, S=Susceptible):\n");
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            printf("%c ", h_population[to1D(i, j, L)].Health == ICU ? 'I' : 'S');
        }
        printf("\n");
    }

    printf("\nInitial bed counts: Hospital=%d, ICU=%d\n", *h_hospitalBeds, *h_icuBeds);

    // Copy to device
    checkCudaError(cudaMemcpy(d_population, h_population, gridSize * sizeof(GPUPerson),
        cudaMemcpyHostToDevice), "Population copy to device");

    // Setup RNG
    setupRNG(gridSize);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (gridSize + blockSize - 1) / blockSize;
    printf("\nLaunching ICU_kernel with %d blocks, %d threads per block\n", numBlocks, blockSize);

    ICU_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L, d_hospitalBeds, d_icuBeds);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    checkCudaError(err, "ICU kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "ICU kernel execution");

    // Copy results back
    checkCudaError(cudaMemcpy(h_population, d_population, gridSize * sizeof(GPUPerson),
        cudaMemcpyDeviceToHost), "Population copy from device");
    checkCudaError(cudaMemcpy(h_hospitalBeds, d_hospitalBeds, sizeof(int),
        cudaMemcpyDeviceToHost), "Hospital beds copy from device");
    checkCudaError(cudaMemcpy(h_icuBeds, d_icuBeds, sizeof(int),
        cudaMemcpyDeviceToHost), "ICU beds copy from device");

    // Print final grid state
    printf("\nFinal Grid State (S=Susceptible, I=ICU, R=Recovered, D=Dead):\n");
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            char state = 'S';
            idx = to1D(i, j, L);
            switch (h_population[idx].Swap) {
            case ICU: state = 'I'; break;
            case Recovered: state = 'R'; break;
            case Dead: state = 'D'; break;
            case DeadCovid: state = 'C'; break;  // C for COVID death
            default: state = 'S'; break;
            }
            printf("%c ", state);
        }
        printf("\n");
    }

    printf("\nFinal bed counts: Hospital=%d, ICU=%d\n", *h_hospitalBeds, *h_icuBeds);

    // Detailed case analysis
    printf("\n=== Detailed Case Results ===\n");

    struct TestCase {
        int i, j;
        const char* description;
        int expectedAge;
    } cases[] = {
        {2, 2, "Young patient (25yo)", 25},
        {3, 3, "Middle-aged patient (55yo)", 55},
        {4, 4, "Elderly patient (75yo)", 75},
        {5, 5, "Very elderly patient (85yo)", 85},
        {6, 6, "Mid-treatment patient (60yo)", 60},
        {7, 7, "Natural death case (70yo)", 70}
    };

    int recoveredCount = 0;
    int diedCovidCount = 0;
    int diedNaturalCount = 0;
    int stillInICUCount = 0;

    for (int c = 0; c < 6; c++) {
        idx = to1D(cases[c].i, cases[c].j, L);
        printf("\nCase %d - %s at (%d,%d):\n", c + 1, cases[c].description, cases[c].i, cases[c].j);
        printf("  Age: %d years\n", h_population[idx].AgeYears);
        printf("  Initial Health: %d\n", h_population[idx].Health);
        printf("  Final State: %d\n", h_population[idx].Swap);
        printf("  TimeOnState: %d\n", h_population[idx].TimeOnState);
        printf("  StateTime: %d\n", h_population[idx].StateTime);
        printf("  Days: %d, AgeDeathDays: %d\n", h_population[idx].Days, h_population[idx].AgeDeathDays);

        const char* outcome = "Unknown";
        switch (h_population[idx].Swap) {
        case Recovered:
            outcome = "RECOVERED";
            recoveredCount++;
            break;
        case DeadCovid:
            outcome = "DIED (COVID)";
            diedCovidCount++;
            break;
        case Dead:
            outcome = "DIED (NATURAL)";
            diedNaturalCount++;
            break;
        case ICU:
            outcome = "STILL IN ICU";
            stillInICUCount++;
            break;
        }
        printf("  Outcome: %s\n", outcome);
    }

    // Summary statistics
    printf("\n=== Summary Statistics ===\n");
    printf("Total ICU patients: 6\n");
    printf("Recovered: %d\n", recoveredCount);
    printf("Died from COVID: %d\n", diedCovidCount);
    printf("Died naturally: %d\n", diedNaturalCount);
    printf("Still in ICU: %d\n", stillInICUCount);

    int bedsFreed = (*h_icuBeds - 8);  // Started with 8, see how many were freed
    printf("ICU beds freed: %d\n", bedsFreed);

    // Expected vs actual analysis
    printf("\n=== Analysis ===\n");
    if (diedNaturalCount > 0) {
        printf("✓ Natural death logic working\n");
    }
    if (stillInICUCount > 0) {
        printf("✓ Time-based treatment continuation working\n");
    }
    if (recoveredCount + diedCovidCount > 0) {
        printf("✓ Recovery/death decision logic working\n");
    }
    if (bedsFreed == (recoveredCount + diedCovidCount + diedNaturalCount)) {
        printf("✓ Bed management working correctly\n");
    }
    else {
        printf("⚠ Bed management issue: expected %d freed, got %d\n",
            recoveredCount + diedCovidCount + diedNaturalCount, bedsFreed);
    }

    // Cleanup
    delete[] h_population;
    delete h_hospitalBeds;
    delete h_icuBeds;
    cudaFree(d_population);
    cudaFree(d_hospitalBeds);
    cudaFree(d_icuBeds);
    cleanupRNG();
    cleanupGPUConstants();

    printf("\n=== Test Completed Successfully ===\n");
}

int main() {
    printf("Starting comprehensive ICU kernel test...\n");
    test_ICU_kernel_comprehensive();
    return 0;
}