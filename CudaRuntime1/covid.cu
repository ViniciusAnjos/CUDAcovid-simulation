

#include <stdio.h>
#include <cuda_runtime.h>
#include "gpu_define.cuh"
#include "gpu_person.cuh"
#include "gpu_utils.cuh"
#include "gpu_aleat.cuh"
#include "gpu_neighbors.cuh"

__global__ void IS_kernel(GPUPerson * population, unsigned int* rngStates, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= L * L) return;

    int i, j;
    to2D(idx, L, i, j);

    printf("Checking position (%d,%d)\n", i, j);

    if (i > 0 && i <= L && j > 0 && j <= L) {
        int personIdx = to1D(i, j, L);
        int health = population[personIdx].Health;

        printf("Position (%d,%d) has Health=%d\n", i, j, health);

        if (health == d_IA || health == d_ISLight ||
            health == d_ISModerate || health == d_ISSevere) {

            printf("Found IS state at (%d,%d): Health=%d\n", i, j, health);

            population[personIdx].TimeOnState++;

            if (population[personIdx].Days >= population[personIdx].AgeDeathDays) {
                population[personIdx].Swap = d_Dead;
                printf("Natural death at (%d,%d)\n", i, j);
            }
            else {
                // Handle IA case
                if (health == d_IA) {
                    printf("Processing IA at (%d,%d)\n", i, j);
                    if (population[personIdx].TimeOnState >= population[personIdx].StateTime) {
                        population[personIdx].Swap = d_Recovered;
                        printf("IA recovered at (%d,%d)\n", i, j);
                    }
                    else {
                        spreadInfection_kernel(i, j, population, &rngStates[idx], L);
                        population[personIdx].Swap = d_IA;
                        printf("IA continues at (%d,%d)\n", i, j);
                    }
                }
                // Handle ISLight case
                else if (health == d_ISLight) {
                    printf("Processing ISLight at (%d,%d)\n", i, j);
                    if (population[personIdx].TimeOnState >= population[personIdx].StateTime) {
                        double rn = generateRandom(&rngStates[idx]);
                        printf("ISLight transition check at (%d,%d): rn=%f vs prob=%f\n",
                            i, j, rn, d_ProbISLightToISModerate);
                        if (rn < d_ProbISLightToISModerate) {
                            population[personIdx].Swap = d_ISModerate;
                            rn = generateRandom(&rngStates[idx]);
                            population[personIdx].StateTime =
                                rn * (d_MaxISModerate - d_MinISModerate) + d_MinISModerate;
                            printf("ISLight -> ISModerate at (%d,%d)\n", i, j);
                        }
                        else {
                            population[personIdx].Swap = d_Recovered;
                            printf("ISLight recovered at (%d,%d)\n", i, j);
                        }
                    }
                    else {
                        spreadInfection_kernel(i, j, population, &rngStates[idx], L);
                        population[personIdx].Swap = d_ISLight;
                        printf("ISLight continues at (%d,%d)\n", i, j);
                    }
                }
                // Handle ISModerate case
                else if (health == d_ISModerate) {
                    printf("Processing ISModerate at (%d,%d)\n", i, j);
                    if (population[personIdx].TimeOnState >= population[personIdx].StateTime) {
                        double rn = generateRandom(&rngStates[idx]);
                        printf("ISModerate recovery check at (%d,%d): rn=%f vs prob=%f\n",
                            i, j, rn, d_ProbRecoveryModerate[population[personIdx].AgeYears]);
                        if (rn < d_ProbRecoveryModerate[population[personIdx].AgeYears]) {
                            population[personIdx].Swap = d_Recovered;
                            printf("ISModerate recovered at (%d,%d)\n", i, j);
                        }
                        else {
                            population[personIdx].Swap = d_ISSevere;
                            rn = generateRandom(&rngStates[idx]);
                            population[personIdx].StateTime =
                                rn * (d_MaxISSevere - d_MinISSevere) + d_MinISSevere;
                            printf("ISModerate -> ISSevere at (%d,%d)\n", i, j);
                        }
                    }
                    else {
                        population[personIdx].Swap = d_ISModerate;
                        printf("ISModerate continues at (%d,%d)\n", i, j);
                    }
                }
                // Handle ISSevere case
                else if (health == d_ISSevere) {
                    printf("Processing ISSevere at (%d,%d)\n", i, j);
                    if (population[personIdx].TimeOnState >= population[personIdx].StateTime) {
                        double rn = generateRandom(&rngStates[idx]);
                        printf("ISSevere recovery check at (%d,%d): rn=%f vs prob=%f\n",
                            i, j, rn, d_ProbRecoverySevere[population[personIdx].AgeYears]);
                        if (rn < d_ProbRecoverySevere[population[personIdx].AgeYears]) {
                            population[personIdx].Swap = d_Recovered;
                            printf("ISSevere recovered at (%d,%d)\n", i, j);
                        }
                        else {
                            if (AvailableBeds > 0) {
                                atomicAdd(&AvailableBeds, -1);
                                population[personIdx].Swap = d_H;
                                rn = generateRandom(&rngStates[idx]);
                                population[personIdx].StateTime =
                                    rn * (d_MaxH - d_MinH) + d_MinH;
                                printf("ISSevere -> Hospital at (%d,%d)\n", i, j);
                            }
                            else {
                                population[personIdx].Swap = d_DeadCovid;
                                printf("ISSevere died (no beds) at (%d,%d)\n", i, j);
                            }
                        }
                    }
                    else {
                        population[personIdx].Swap = d_ISSevere;
                        printf("ISSevere continues at (%d,%d)\n", i, j);
                    }
                }
            }
            printf("Final Swap state for (%d,%d): %d\n", i, j, population[personIdx].Swap);
        }
        else {
            printf("Position (%d,%d) not an IS state (Health=%d)\n", i, j, health);
        }
    }
}


void test_IS_state() {
    const int L = 7;
    const int gridDim = (L + 2) * (L + 2);

    printf("\nVerifying State Constants:\n");
    printf("Original values: S=%d, IA=%d, ISLight=%d, ISModerate=%d, ISSevere=%d, Recovered=%d\n",
        S, IA, ISLight, ISModerate, ISSevere, Recovered);

    // Verify GPU constants
    int d_s, d_ia, d_islight, d_ismoderate, d_issevere, d_recovered;
    cudaMemcpyFromSymbol(&d_s, d_S, sizeof(int));
    cudaMemcpyFromSymbol(&d_ia, d_IA, sizeof(int));
    cudaMemcpyFromSymbol(&d_islight, d_ISLight, sizeof(int));
    cudaMemcpyFromSymbol(&d_ismoderate, d_ISModerate, sizeof(int));
    cudaMemcpyFromSymbol(&d_issevere, d_ISSevere, sizeof(int));
    cudaMemcpyFromSymbol(&d_recovered, d_Recovered, sizeof(int));

    printf("GPU values: S=%d, IA=%d, ISLight=%d, ISModerate=%d, ISSevere=%d, Recovered=%d\n",
        d_s, d_ia, d_islight, d_ismoderate, d_issevere, d_recovered);

    // Verify initial available beds
    int verify_beds;
    cudaMemcpyFromSymbol(&verify_beds, AvailableBeds, sizeof(int));
    printf("\nInitial Available Beds: %d\n", verify_beds);

    // Allocate and initialize test population
    GPUPerson* h_population = new GPUPerson[gridDim];
    GPUPerson* d_population;
    cudaMalloc(&d_population, gridDim * sizeof(GPUPerson));

    // Initialize all as Susceptible
    printf("\nInitializing population...\n");
    for (int i = 0; i < gridDim; i++) {
        h_population[i].Health = S;
        h_population[i].Swap = S;
        h_population[i].TimeOnState = 0;
        h_population[i].StateTime = 0;
        h_population[i].Days = 0;
        h_population[i].AgeYears = 40;
        h_population[i].AgeDeathDays = 365 * 80;
        h_population[i].Checked = 0;
        h_population[i].Exponent = 0;
    }

    printf("\nTest Case Setup Details:\n");

    // Case 1: Asymptomatic about to recover
    int idx = to1D(2, 2, L);
    h_population[idx].Health = IA;
    h_population[idx].StateTime = 5;
    h_population[idx].TimeOnState = 5;
    printf("\nCase 1 (IA) - Initial setup at (%d,%d):\n", 2, 2);
    printf("- Health: %d (should be %d)\n", h_population[idx].Health, IA);
    printf("- StateTime: %d\n", h_population[idx].StateTime);
    printf("- TimeOnState: %d\n", h_population[idx].TimeOnState);

    // Case 2: Light symptoms about to progress
    idx = to1D(3, 3, L);
    h_population[idx].Health = ISLight;
    h_population[idx].StateTime = 5;
    h_population[idx].TimeOnState = 5;
    printf("\nCase 2 (ISLight) - Initial setup at (%d,%d):\n", 3, 3);
    printf("- Health: %d (should be %d)\n", h_population[idx].Health, ISLight);
    printf("- StateTime: %d\n", h_population[idx].StateTime);
    printf("- TimeOnState: %d\n", h_population[idx].TimeOnState);

    // Case 3: Moderate symptoms (young)
    idx = to1D(4, 4, L);
    h_population[idx].Health = ISModerate;
    h_population[idx].StateTime = 5;
    h_population[idx].TimeOnState = 5;
    h_population[idx].AgeYears = 25;
    printf("\nCase 3 (ISModerate-Young) - Initial setup at (%d,%d):\n", 4, 4);
    printf("- Health: %d (should be %d)\n", h_population[idx].Health, ISModerate);
    printf("- StateTime: %d\n", h_population[idx].StateTime);
    printf("- TimeOnState: %d\n", h_population[idx].TimeOnState);
    printf("- Age: %d\n", h_population[idx].AgeYears);

    // Case 4: Moderate symptoms (elderly)
    idx = to1D(5, 5, L);
    h_population[idx].Health = ISModerate;
    h_population[idx].StateTime = 5;
    h_population[idx].TimeOnState = 5;
    h_population[idx].AgeYears = 75;
    printf("\nCase 4 (ISModerate-Elderly) - Initial setup at (%d,%d):\n", 5, 5);
    printf("- Health: %d (should be %d)\n", h_population[idx].Health, ISModerate);
    printf("- StateTime: %d\n", h_population[idx].StateTime);
    printf("- TimeOnState: %d\n", h_population[idx].TimeOnState);
    printf("- Age: %d\n", h_population[idx].AgeYears);

    // Case 5: Severe symptoms
    idx = to1D(2, 4, L);
    h_population[idx].Health = ISSevere;
    h_population[idx].StateTime = 5;
    h_population[idx].TimeOnState = 5;
    printf("\nCase 5 (ISSevere) - Initial setup at (%d,%d):\n", 2, 4);
    printf("- Health: %d (should be %d)\n", h_population[idx].Health, ISSevere);
    printf("- StateTime: %d\n", h_population[idx].StateTime);
    printf("- TimeOnState: %d\n", h_population[idx].TimeOnState);

    // Print transition probabilities
    double verify_probISLightToISModerate;
    cudaMemcpyFromSymbol(&verify_probISLightToISModerate, d_ProbISLightToISModerate, sizeof(double));
    printf("\nTransition Probabilities:\n");
    printf("ProbISLightToISModerate: %f\n", verify_probISLightToISModerate);
    printf("Available Hospital Beds: %d\n", verify_beds);

    // Print initial grid
    printf("\nInitial Grid State:\n");
    printf("S: Susceptible, A: Asymptomatic, L: Light, M: Moderate, V: Severe\n");
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            char state = 'S';
            idx = to1D(i, j, L);
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
    cudaMemcpy(d_population, h_population, gridDim * sizeof(GPUPerson),
        cudaMemcpyHostToDevice);

    setupRNG(gridDim);

    int blockSize = 256;
    int numBlocks = (gridDim + blockSize - 1) / blockSize;
    printf("\nLaunching IS_kernel with %d blocks, %d threads per block\n",
        numBlocks, blockSize);

    printf("\nGrid Configuration:\n");
    printf("L: %d\n", L);
    printf("Total Grid Size (with boundaries): %d x %d\n", L + 2, L + 2);
    printf("Block Size: %d\n", blockSize);
    printf("Number of Blocks: %d\n", numBlocks);

    IS_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);
    cudaDeviceSynchronize();

    // Get final bed count
    cudaMemcpyFromSymbol(&verify_beds, AvailableBeds, sizeof(int));
    printf("\nFinal Available Beds: %d\n", verify_beds);

    // Get results
    cudaMemcpy(h_population, d_population, gridDim * sizeof(GPUPerson),
        cudaMemcpyDeviceToHost);

    // Print final state
    printf("\nFinal Grid State:\n");
    printf("S: Susceptible, A: Asymptomatic, L: Light, M: Moderate\n");
    printf("V: Severe, H: Hospitalized, R: Recovered, D: Dead\n");
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            char state = 'S';
            idx = to1D(i, j, L);
            switch (h_population[idx].Swap) {
            case IA: state = 'A'; break;
            case ISLight: state = 'L'; break;
            case ISModerate: state = 'M'; break;
            case ISSevere: state = 'V'; break;
            case H: state = 'H'; break;
            case Recovered: state = 'R'; break;
            case Dead: case DeadCovid: state = 'D'; break;
            }
            printf("%c ", state);
        }
        printf("\n");
    }

    // Print detailed case results
    printf("\nDetailed Final State for Test Cases:\n");

    printf("\nCase 1 (2,2) Final State:\n");
    idx = to1D(2, 2, L);
    printf("- Initial Health: %d\n", h_population[idx].Health);
    printf("- Final Swap: %d\n", h_population[idx].Swap);
    printf("- TimeOnState: %d\n", h_population[idx].TimeOnState);
    printf("- StateTime: %d\n", h_population[idx].StateTime);
    printf("- Age: %d\n", h_population[idx].AgeYears);

    printf("\nCase 2 (3,3) Final State:\n");
    idx = to1D(3, 3, L);
    printf("- Initial Health: %d\n", h_population[idx].Health);
    printf("- Final Swap: %d\n", h_population[idx].Swap);
    printf("- TimeOnState: %d\n", h_population[idx].TimeOnState);
    printf("- StateTime: %d\n", h_population[idx].StateTime);
    printf("- Age: %d\n", h_population[idx].AgeYears);

    printf("\nCase 3 (4,4) Final State:\n");
    idx = to1D(4, 4, L);
    printf("- Initial Health: %d\n", h_population[idx].Health);
    printf("- Final Swap: %d\n", h_population[idx].Swap);
    printf("- TimeOnState: %d\n", h_population[idx].TimeOnState);
    printf("- StateTime: %d\n", h_population[idx].StateTime);
    printf("- Age: %d\n", h_population[idx].AgeYears);

    printf("\nCase 4 (5,5) Final State:\n");
    idx = to1D(5, 5, L);
    printf("- Initial Health: %d\n", h_population[idx].Health);
    printf("- Final Swap: %d\n", h_population[idx].Swap);
    printf("- TimeOnState: %d\n", h_population[idx].TimeOnState);
    printf("- StateTime: %d\n", h_population[idx].StateTime);
    printf("- Age: %d\n", h_population[idx].AgeYears);

    printf("\nCase 5 (2,4) Final State:\n");
    idx = to1D(2, 4, L);
    printf("- Initial Health: %d\n", h_population[idx].Health);
    printf("- Final Swap: %d\n", h_population[idx].Swap);
    printf("- TimeOnState: %d\n", h_population[idx].TimeOnState);
    printf("- StateTime: %d\n", h_population[idx].StateTime);
    printf("- Age: %d\n", h_population[idx].AgeYears);

    printf("\nFinal Available Hospital Beds: %d\n", verify_beds);

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