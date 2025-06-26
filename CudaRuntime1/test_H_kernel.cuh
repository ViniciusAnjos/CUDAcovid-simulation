#include <stdio.h>
#include <cuda_runtime.h>
#include "gpu_define.cuh"
#include "gpu_person.cuh"
#include "gpu_utils.cuh"
#include "gpu_aleat.cuh"

// Include the H kernel
#include "H_kernel.cuh"


void test_H_kernel_comprehensive() {
    const int L = 10;  // 10x10 grid for multiple test cases
    const int gridSize = (L + 2) * (L + 2);

    printf("=== Comprehensive H Kernel Test ===\n");
    printf("Grid size: %dx%d = %d total cells\n", L, L, L * L);

    // Setup GPU constants
    int city = ROC;  // Use Rocinha parameters
    setupCityParameters(city);
    setupGPUConstants();

    // Allocate memory
    GPUPerson* h_population = new GPUPerson[gridSize];
    GPUPerson* d_population;
    checkCudaError(cudaMalloc(&d_population, gridSize * sizeof(GPUPerson)), "Population allocation");

    // Bed counters - start with limited beds to test resource constraints
    int* h_hospitalBeds = new int(6);   // Start with 6 hospital beds
    int* h_icuBeds = new int(3);        // Start with 3 ICU beds
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

    // Case 1: Young hospitalized patient (high recovery chance)
    printf("Case 1: Young hospital patient (25yo) ready to transition at (2,2)\n");
    int idx = to1D(2, 2, L);
    h_population[idx].Health = H;
    h_population[idx].Swap = H;
    h_population[idx].AgeYears = 25;
    h_population[idx].StateTime = 10;
    h_population[idx].TimeOnState = 10;  // Ready to transition
    h_population[idx].Days = 25 * 365;   // 25 years old

    // Case 2: Middle-aged hospital patient (good recovery chance)
    printf("Case 2: Middle-aged hospital patient (45yo) ready to transition at (3,3)\n");
    idx = to1D(3, 3, L);
    h_population[idx].Health = H;
    h_population[idx].Swap = H;
    h_population[idx].AgeYears = 45;
    h_population[idx].StateTime = 15;
    h_population[idx].TimeOnState = 15;  // Ready to transition
    h_population[idx].Days = 45 * 365;

    // Case 3: Elderly hospital patient (lower recovery chance)
    printf("Case 3: Elderly hospital patient (75yo) ready to transition at (4,4)\n");
    idx = to1D(4, 4, L);
    h_population[idx].Health = H;
    h_population[idx].Swap = H;
    h_population[idx].AgeYears = 75;
    h_population[idx].StateTime = 12;
    h_population[idx].TimeOnState = 12;  // Ready to transition
    h_population[idx].Days = 75 * 365;

    // Case 4: Very elderly hospital patient (low recovery chance)
    printf("Case 4: Very elderly hospital patient (85yo) ready to transition at (5,5)\n");
    idx = to1D(5, 5, L);
    h_population[idx].Health = H;
    h_population[idx].Swap = H;
    h_population[idx].AgeYears = 85;
    h_population[idx].StateTime = 8;
    h_population[idx].TimeOnState = 8;   // Ready to transition
    h_population[idx].Days = 85 * 365;

    // Case 5: Another elderly patient (to test ICU bed shortage)
    printf("Case 5: Another elderly hospital patient (80yo) ready to transition at (6,6)\n");
    idx = to1D(6, 6, L);
    h_population[idx].Health = H;
    h_population[idx].Swap = H;
    h_population[idx].AgeYears = 80;
    h_population[idx].StateTime = 10;
    h_population[idx].TimeOnState = 10;  // Ready to transition
    h_population[idx].Days = 80 * 365;

    // Case 6: Patient in middle of hospital stay
    printf("Case 6: Hospital patient (60yo) in middle of treatment at (7,7)\n");
    idx = to1D(7, 7, L);
    h_population[idx].Health = H;
    h_population[idx].Swap = H;
    h_population[idx].AgeYears = 60;
    h_population[idx].StateTime = 20;
    h_population[idx].TimeOnState = 5;   // Not ready to transition
    h_population[idx].Days = 60 * 365;

    // Case 7: Natural death case
    printf("Case 7: Hospital patient (70yo) reaching natural death at (8,8)\n");
    idx = to1D(8, 8, L);
    h_population[idx].Health = H;
    h_population[idx].Swap = H;
    h_population[idx].AgeYears = 70;
    h_population[idx].StateTime = 10;
    h_population[idx].TimeOnState = 5;
    h_population[idx].Days = 85 * 365;     // 85 years in days
    h_population[idx].AgeDeathDays = 80 * 365;  // Should die at 80

    // Case 8: Another patient to test resource limits
    printf("Case 8: Another elderly hospital patient (78yo) ready to transition at (9,9)\n");
    idx = to1D(9, 9, L);
    h_population[idx].Health = H;
    h_population[idx].Swap = H;
    h_population[idx].AgeYears = 78;
    h_population[idx].StateTime = 14;
    h_population[idx].TimeOnState = 14;  // Ready to transition
    h_population[idx].Days = 78 * 365;

    // Print recovery probabilities for reference
    printf("\nExpected Recovery Probabilities (from define.h):\n");
    printf("Age 25: ~96.2%% (ProbRecoveryH_20_30)\n");
    printf("Age 45: ~93.8%% (ProbRecoveryH_40_50)\n");
    printf("Age 75: ~67.8%% (ProbRecoveryH_70_80)\n");
    printf("Age 85: ~45.7%% (ProbRecoveryH_80_90)\n");

    // Print initial grid state
    printf("\nInitial Grid State (H=Hospital, S=Susceptible):\n");
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            printf("%c ", h_population[to1D(i, j, L)].Health == H ? 'H' : 'S');
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
    printf("\nLaunching H_kernel with %d blocks, %d threads per block\n", numBlocks, blockSize);

    H_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L, d_hospitalBeds, d_icuBeds);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    checkCudaError(err, "H kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "H kernel execution");

    // Copy results back
    checkCudaError(cudaMemcpy(h_population, d_population, gridSize * sizeof(GPUPerson),
        cudaMemcpyDeviceToHost), "Population copy from device");
    checkCudaError(cudaMemcpy(h_hospitalBeds, d_hospitalBeds, sizeof(int),
        cudaMemcpyDeviceToHost), "Hospital beds copy from device");
    checkCudaError(cudaMemcpy(h_icuBeds, d_icuBeds, sizeof(int),
        cudaMemcpyDeviceToHost), "ICU beds copy from device");

    // Print final grid state
    printf("\nFinal Grid State (S=Susceptible, H=Hospital, R=Recovered, I=ICU, D=Dead, C=COVID Death):\n");
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            char state = 'S';
            idx = to1D(i, j, L);
            switch (h_population[idx].Swap) {
            case H: state = 'H'; break;
            case ICU: state = 'I'; break;
            case Recovered: state = 'R'; break;
            case Dead: state = 'D'; break;
            case DeadCovid: state = 'C'; break;
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
        {3, 3, "Middle-aged patient (45yo)", 45},
        {4, 4, "Elderly patient (75yo)", 75},
        {5, 5, "Very elderly patient (85yo)", 85},
        {6, 6, "Another elderly patient (80yo)", 80},
        {7, 7, "Mid-treatment patient (60yo)", 60},
        {8, 8, "Natural death case (70yo)", 70},
        {9, 9, "Another elderly patient (78yo)", 78}
    };

    int recoveredCount = 0;
    int diedCovidCount = 0;
    int diedNaturalCount = 0;
    int transferredICUCount = 0;
    int stillInHospitalCount = 0;

    for (int c = 0; c < 8; c++) {
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
        case ICU:
            outcome = "TRANSFERRED TO ICU";
            transferredICUCount++;
            break;
        case DeadCovid:
            outcome = "DIED (COVID - No ICU)";
            diedCovidCount++;
            break;
        case Dead:
            outcome = "DIED (NATURAL)";
            diedNaturalCount++;
            break;
        case H:
            outcome = "STILL IN HOSPITAL";
            stillInHospitalCount++;
            break;
        }
        printf("  Outcome: %s\n", outcome);
    }

    // Summary statistics
    printf("\n=== Summary Statistics ===\n");
    printf("Total hospital patients: 8\n");
    printf("Recovered: %d\n", recoveredCount);
    printf("Transferred to ICU: %d\n", transferredICUCount);
    printf("Died from COVID (no ICU bed): %d\n", diedCovidCount);
    printf("Died naturally: %d\n", diedNaturalCount);
    printf("Still in hospital: %d\n", stillInHospitalCount);

    int hospitalBedsFreed = (*h_hospitalBeds - 6);  // Started with 6
    int icuBedsUsed = (3 - *h_icuBeds);             // Started with 3
    printf("Hospital beds freed: %d\n", hospitalBedsFreed);
    printf("ICU beds used: %d\n", icuBedsUsed);

    // Expected vs actual analysis
    printf("\n=== Analysis ===\n");
    if (diedNaturalCount > 0) {
        printf("✓ Natural death logic working\n");
    }
    if (stillInHospitalCount > 0) {
        printf("✓ Time-based treatment continuation working\n");
    }
    if (recoveredCount + transferredICUCount + diedCovidCount > 0) {
        printf("✓ Recovery/ICU/death decision logic working\n");
    }
    if (transferredICUCount == icuBedsUsed) {
        printf("✓ ICU bed allocation working correctly\n");
    }
    if (diedCovidCount > 0) {
        printf("✓ ICU bed shortage handling working (patients die when no ICU beds)\n");
    }

    // Expected bed management
    int expectedFreed = recoveredCount + transferredICUCount + diedCovidCount + diedNaturalCount;
    if (hospitalBedsFreed == expectedFreed) {
        printf("✓ Hospital bed management working correctly\n");
    }
    else {
        printf("⚠ Hospital bed management issue: expected %d freed, got %d\n",
            expectedFreed, hospitalBedsFreed);
    }

    // Age-based outcome analysis
    printf("\n=== Age-Based Outcome Analysis ===\n");
    printf("Young patients (25-45): Recovery expected\n");
    printf("Elderly patients (75-85): Lower recovery, ICU transfer or death expected\n");
    printf("Resource constraints: With only 3 ICU beds, some elderly may die from lack of beds\n");

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
    printf("Starting comprehensive H kernel test...\n");
    test_H_kernel_comprehensive();
    return 0;
}