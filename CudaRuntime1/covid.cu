#include <stdio.h>
#include <cuda_runtime.h>
#include "gpu_define.cuh"
#include "gpu_person.cuh"
#include "gpu_utils.cuh"
#include "gpu_aleat.cuh"
#include "update_kernel.cuh"

void printPopulationState(GPUPerson* h_population, int L, const char* title) {
    printf("\n%s:\n", title);
    printf("Grid visualization (S=Susceptible, E=Exposed, I=Infectious, etc.):\n");

    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            int idx = to1D(i, j, L);
            char symbol = '?';

            int health = h_population[idx].Health;
            if (health == S) symbol = 'S';
            else if (health == E) symbol = 'E';
            else if (health == IP) symbol = 'P';
            else if (health == IA) symbol = 'A';
            else if (health == ISLight) symbol = 'L';
            else if (health == ISModerate) symbol = 'M';
            else if (health == ISSevere) symbol = 'V';
            else if (health == H) symbol = 'H';
            else if (health == ICU) symbol = 'U';
            else if (health == Recovered) symbol = 'R';
            else if (health == Dead) symbol = 'D';
            else if (health == DeadCovid) symbol = 'C';

            printf("%c ", symbol);
        }
        printf("\n");
    }
}

void test_update_kernel() {
    const int L = 10;  // Small grid for testing
    const int gridSize = (L + 2) * (L + 2);

    // Allocate and initialize test population
    GPUPerson* h_population = new GPUPerson[gridSize];
    GPUPerson* d_population;
    cudaMalloc(&d_population, gridSize * sizeof(GPUPerson));

    printf("Testing Update Kernel with %dx%d grid\n", L, L);
    printf("Now using UNIFORM age distribution (0-100) matching original code\n\n");

    // Initialize all cells
    for (int i = 0; i < gridSize; i++) {
        h_population[i].Health = S;
        h_population[i].Swap = S;
        h_population[i].TimeOnState = 0;
        h_population[i].StateTime = 0;
        h_population[i].Days = 0;
        h_population[i].AgeDays = 365 * 30;  // 30 years old
        h_population[i].AgeYears = 30;
        h_population[i].AgeDeathDays = 365 * 80;  // Dies at 80
        h_population[i].AgeDeathYears = 80;
        h_population[i].Checked = 0;
        h_population[i].Exponent = 0;
        h_population[i].Isolation = IsolationNo;
    }

    // Set up test cases
    printf("Setting up test cases:\n");

    // Test Case 1: State transitions (set different Swap values)
    printf("Case 1: Person at (2,2) transitions S->E\n");
    h_population[to1D(2, 2, L)].Swap = E;
    h_population[to1D(2, 2, L)].StateTime = 5;

    printf("Case 2: Person at (3,3) transitions E->IP\n");
    h_population[to1D(3, 3, L)].Health = E;
    h_population[to1D(3, 3, L)].Swap = IP;

    printf("Case 3: Person at (4,4) recovers\n");
    h_population[to1D(4, 4, L)].Health = ISLight;
    h_population[to1D(4, 4, L)].Swap = Recovered;

    // Test Case 4: Natural death
    printf("Case 4: Person at (5,5) dies naturally\n");
    h_population[to1D(5, 5, L)].Days = 365 * 85;  // Over death age
    h_population[to1D(5, 5, L)].AgeDays = 365 * 85;

    // Test Case 5: COVID death replacement
    printf("Case 5: Person at (6,6) died from COVID (to be replaced)\n");
    h_population[to1D(6, 6, L)].Health = DeadCovid;
    h_population[to1D(6, 6, L)].Swap = DeadCovid;

    // Test Case 6: Test exponent/checked reset
    printf("Case 6: Person at (7,7) has exposure counters to reset\n");
    h_population[to1D(7, 7, L)].Exponent = 3;
    h_population[to1D(7, 7, L)].Checked = 1;

    // Test boundary conditions setup
    printf("Case 7: Boundary test - person at (1,1) for corner check\n");
    h_population[to1D(1, 1, L)].Health = IP;
    h_population[to1D(1, 1, L)].Swap = IP;

    h_population[to1D(L, L, L)].Health = IA;
    h_population[to1D(L, L, L)].Swap = IA;

    // Add multiple death cases to test age distribution
    printf("\nAdding multiple death cases to test uniform age distribution:\n");
    for (int i = 2; i <= 5; i++) {
        printf("Case %d: Person at (8,%d) died from COVID\n", 6 + i, i);
        h_population[to1D(8, i, L)].Health = DeadCovid;
        h_population[to1D(8, i, L)].Swap = DeadCovid;
    }

    // Copy to device
    cudaMemcpy(d_population, h_population, gridSize * sizeof(GPUPerson),
        cudaMemcpyHostToDevice);

    // Setup RNG
    setupRNG(gridSize);

    // Print initial state
    printPopulationState(h_population, L, "Initial State");

    // Reset counters
    resetCounters_kernel << <1, 1 >> > ();
    resetNewCounters_kernel << <1, 1 >> > ();
    cudaDeviceSynchronize();

    // First update boundaries
    int boundaryBlocks = (L + 255) / 256;
    updateBoundaries_kernel << <boundaryBlocks, 256 >> > (d_population, L);

    // Then run update kernel - NOTE: Simplified parameters (no age distribution arrays)
    int blockSize = 256;
    int numBlocks = (gridSize + blockSize - 1) / blockSize;

    printf("\nRunning update_kernel with uniform age distribution...\n");
    update_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L, 1,
        d_ProbNaturalDeath);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(h_population, d_population, gridSize * sizeof(GPUPerson),
        cudaMemcpyDeviceToHost);

    // Print final state
    printPopulationState(h_population, L, "Final State After Update");

    // Get counters
    int h_totals[15] = { 0 };
    int h_new_cases[15] = { 0 };
    getCountersFromDevice(h_totals, h_new_cases);

    // Print statistics
    printf("\nPopulation Statistics:\n");
    printf("S: %d (new: %d)\n", h_totals[S], h_new_cases[S]);
    printf("E: %d (new: %d)\n", h_totals[E], h_new_cases[E]);
    printf("IP: %d (new: %d)\n", h_totals[IP], h_new_cases[IP]);
    printf("IA: %d (new: %d)\n", h_totals[IA], h_new_cases[IA]);
    printf("ISLight: %d (new: %d)\n", h_totals[ISLight], h_new_cases[ISLight]);
    printf("ISModerate: %d (new: %d)\n", h_totals[ISModerate], h_new_cases[ISModerate]);
    printf("ISSevere: %d (new: %d)\n", h_totals[ISSevere], h_new_cases[ISSevere]);
    printf("H: %d (new: %d)\n", h_totals[H], h_new_cases[H]);
    printf("ICU: %d (new: %d)\n", h_totals[ICU], h_new_cases[ICU]);
    printf("Recovered: %d (new: %d)\n", h_totals[Recovered], h_new_cases[Recovered]);
    printf("DeadCovid: %d (new: %d)\n", h_totals[DeadCovid], h_new_cases[DeadCovid]);
    printf("Dead: %d (new: %d)\n", h_totals[Dead], h_new_cases[Dead]);

    // Verify specific test cases
    printf("\nTest Case Verification:\n");

    printf("Case 1 (2,2): Health should be E = %d (actual: %d)\n",
        E, h_population[to1D(2, 2, L)].Health);

    printf("Case 2 (3,3): Health should be IP = %d (actual: %d)\n",
        IP, h_population[to1D(3, 3, L)].Health);

    printf("Case 3 (4,4): Health should be Recovered = %d (actual: %d)\n",
        Recovered, h_population[to1D(4, 4, L)].Health);

    printf("Case 4 (5,5): Health should be Dead = %d (actual: %d)\n",
        Dead, h_population[to1D(5, 5, L)].Health);

    printf("Case 5 (6,6): Health should be S = %d (actual: %d) - Replaced individual\n",
        S, h_population[to1D(6, 6, L)].Health);
    printf("  New age: %d years, Death age: %d years\n",
        h_population[to1D(6, 6, L)].AgeYears,
        h_population[to1D(6, 6, L)].AgeDeathYears);

    printf("Case 6 (7,7): Exponent should be 0 (actual: %d), Checked should be 0 (actual: %d)\n",
        h_population[to1D(7, 7, L)].Exponent,
        h_population[to1D(7, 7, L)].Checked);

    // Check boundaries
    printf("\nBoundary verification:\n");
    printf("Corner (0,0) should match (L,L): %d == %d\n",
        h_population[to1D(0, 0, L)].Health,
        h_population[to1D(L, L, L)].Health);

    // Show ages of all replaced individuals to verify uniform distribution
    printf("\nAge Distribution of Replaced Individuals (should be uniform 0-100):\n");
    printf("Case 5 (6,6): Age = %d years\n", h_population[to1D(6, 6, L)].AgeYears);
    printf("Case 4 (5,5): Age = %d years (natural death replacement)\n",
        h_population[to1D(5, 5, L)].AgeYears);

    // Show ages from multiple death replacements
    for (int i = 2; i <= 5; i++) {
        printf("Person at (8,%d): Age = %d years\n",
            i, h_population[to1D(8, i, L)].AgeYears);
    }

    printf("\nNote: With uniform distribution, ages can be anywhere from 0-100 with equal probability\n");
    printf("(Unlike demographic distribution where young ages would be more common)\n");

    // Cleanup
    delete[] h_population;
    cudaFree(d_population);
    cleanupRNG();
}

int main() {
    // Initialize city and GPU constants
    int city = ROC;
    setupCityParameters(city);
    setupGPUConstants();

    // Run the test
    test_update_kernel();

    // Cleanup
    cleanupGPUConstants();

    return 0;
}