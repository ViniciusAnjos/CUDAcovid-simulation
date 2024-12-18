#include <stdio.h>
#include <cuda_runtime.h>
#include "gpu_define.cuh"
#include "gpu_person.cuh"
#include "gpu_utils.cuh"
#include "gpu_aleat.cuh"
#include "gpu_neighbors.cuh"

#include "IS_kernel.cuh"


void test_IS_state() {
    const int L = 7;  // Larger grid to accommodate all test cases
    const int gridSize = (L + 2) * (L + 2);

    // Allocate and initialize test population
    GPUPerson* h_population = new GPUPerson[gridSize];
    GPUPerson* d_population;
    cudaMalloc(&d_population, gridSize * sizeof(GPUPerson));

    // Initialize all as Susceptible
    printf("\nInitializing population...\n");
    for (int i = 0; i < gridSize; i++) {
        h_population[i].Health = S;
        h_population[i].Swap = S;
        h_population[i].TimeOnState = 0;
        h_population[i].StateTime = 0;
        h_population[i].Days = 0;
        h_population[i].AgeYears = 40;  // Middle-aged by default
        h_population[i].AgeDeathDays = 365 * 80;
        h_population[i].Checked = 0;
        h_population[i].Exponent = 0;
    }

    // Set up test cases
    printf("\nSetting up test cases:\n");

    // Case 1: Asymptomatic about to recover
    printf("\nCase 1 - Asymptomatic near recovery (2,2):\n");
    h_population[to1D(2, 2, L)].Health = IA;
    h_population[to1D(2, 2, L)].StateTime = 5;
    h_population[to1D(2, 2, L)].TimeOnState = 5;

    // Case 2: Light symptoms about to progress
    printf("\nCase 2 - Light symptoms case (3,3):\n");
    h_population[to1D(3, 3, L)].Health = ISLight;
    h_population[to1D(3, 3, L)].StateTime = 5;
    h_population[to1D(3, 3, L)].TimeOnState = 5;

    // Case 3: Moderate symptoms (young)
    printf("\nCase 3 - Moderate symptoms young case (4,4):\n");
    h_population[to1D(4, 4, L)].Health = ISModerate;
    h_population[to1D(4, 4, L)].StateTime = 5;
    h_population[to1D(4, 4, L)].TimeOnState = 5;
    h_population[to1D(4, 4, L)].AgeYears = 25;

    // Case 4: Moderate symptoms (elderly)
    printf("\nCase 4 - Moderate symptoms elderly case (5,5):\n");
    h_population[to1D(5, 5, L)].Health = ISModerate;
    h_population[to1D(5, 5, L)].StateTime = 5;
    h_population[to1D(5, 5, L)].TimeOnState = 5;
    h_population[to1D(5, 5, L)].AgeYears = 75;

    // Case 5: Severe symptoms with available beds
    printf("\nCase 5 - Severe symptoms with beds (2,4):\n");
    h_population[to1D(2, 4, L)].Health = ISSevere;
    h_population[to1D(2, 4, L)].StateTime = 5;
    h_population[to1D(2, 4, L)].TimeOnState = 5;

    // Print probabilities
    printf("\nTransition Probabilities:\n");
    printf("ProbISLightToISModerate: %f\n", ProbISLightToISModerate);
    printf("Available Hospital Beds: %d\n", AvailableBeds);

    // Print initial grid
    printf("\nInitial Grid State:\n");
    printf("S: Susceptible, A: Asymptomatic, L: Light, M: Moderate, V: Severe\n");
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            char state = 'S';
            int idx = to1D(i, j, L);
            switch (h_population[idx].Health) {
            case IA: state = 'A'; break;
            case ISLight: state = 'L'; break;
            case ISModerate: state = 'M'; break;
            case ISSevere: state = 'V'; break;
            }
            printf("%c ", state);
        }
        printf("\n");
    }

    // Copy to device and run kernel
    cudaMemcpy(d_population, h_population, gridSize * sizeof(GPUPerson),
        cudaMemcpyHostToDevice);

    setupRNG(gridSize);

    int blockSize = 256;
    int numBlocks = (gridSize + blockSize - 1) / blockSize;
    printf("\nLaunching IS_kernel with %d blocks, %d threads per block\n",
        numBlocks, blockSize);

    IS_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);

    // Get results
    cudaMemcpy(h_population, d_population, gridSize * sizeof(GPUPerson),
        cudaMemcpyDeviceToHost);

    // Print final state
    printf("\nFinal Grid State:\n");
    printf("S: Susceptible, A: Asymptomatic, L: Light, M: Moderate\n");
    printf("V: Severe, H: Hospitalized, R: Recovered, D: Dead\n");
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            char state = 'S';
            int idx = to1D(i, j, L);
            switch (h_population[idx].Swap) {
            case IA: state = 'A'; break;
            case ISLight: state = 'L'; break;
            case ISModerate: state = 'M'; break;
            case ISSevere: state = 'V'; break;
            case H: state = 'H'; break;
            case Recovered: state = 'R'; break;
            case Dead: state = 'D'; break;
            case DeadCovid: state = 'D'; break;
            }
            printf("%c ", state);
        }
        printf("\n");
    }

    // Print detailed case results
    printf("\nDetailed Final State for Test Cases:\n");
    for (int caseNum = 1; caseNum <= 5; caseNum++) {
        int i, j;
        switch (caseNum) {
        case 1: i = 2; j = 2; break;  // Asymptomatic
        case 2: i = 3; j = 3; break;  // Light
        case 3: i = 4; j = 4; break;  // Moderate (young)
        case 4: i = 5; j = 5; break;  // Moderate (elderly)
        case 5: i = 2; j = 4; break;  // Severe
        }

        int idx = to1D(i, j, L);
        printf("\nCase %d (%d,%d) Final State:\n", caseNum, i, j);
        printf("- Initial Health: %d\n", h_population[idx].Health);
        printf("- Final Swap: %d\n", h_population[idx].Swap);
        printf("- TimeOnState: %d\n", h_population[idx].TimeOnState);
        printf("- StateTime: %d\n", h_population[idx].StateTime);
        printf("- Age: %d\n", h_population[idx].AgeYears);
    }

    printf("\nFinal Available Hospital Beds: %d\n", AvailableBeds);

    // Cleanup
    delete[] h_population;
    cudaFree(d_population);
    cleanupRNG();
}

int main() {
    int city = ROC;
    setupCityParameters(city);
    setupGPUConstants();

    test_IS_state();

    cleanupGPUConstants();
    return 0;
}