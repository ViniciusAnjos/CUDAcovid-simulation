// Enhanced ICU_kernel with debug output
__global__ void ICU_kernel(GPUPerson* population, unsigned int* rngStates, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * L) return;

    int i, j;
    to2D(idx, L, i, j);

    if (i > 0 && i <= L && j > 0 && j <= L) {
        int personIdx = to1D(i, j, L);
        if (population[personIdx].Health == d_ICU) {
            printf("\nProcessing ICU case at (%d,%d):\n", i, j);
            printf("- Initial: TimeOnState=%d, StateTime=%d, Days=%d, Age=%d\n",
                population[personIdx].TimeOnState,
                population[personIdx].StateTime,
                population[personIdx].Days,
                population[personIdx].AgeYears);

            population[personIdx].TimeOnState++;

            if (population[personIdx].Days >= population[personIdx].AgeDeathDays) {
                population[personIdx].Swap = d_Dead;
                atomicAdd(&AvailableBedsICU, 1);
                printf("- Natural death occurred, freed ICU bed\n");
            }
            else {
                if (population[personIdx].TimeOnState >= population[personIdx].StateTime) {
                    double rn = generateRandom(&rngStates[idx]);
                    double recoveryProb = d_ProbRecoveryICU[population[personIdx].AgeYears];
                    printf("- Time in ICU complete, RN=%f (recovery prob=%f)\n",
                        rn, recoveryProb);

                    if (rn < recoveryProb) {
                        population[personIdx].Swap = d_Recovered;
                        atomicAdd(&AvailableBedsICU, 1);
                        printf("- Recovered from ICU, freed bed\n");
                    }
                    else {
                        population[personIdx].Swap = d_DeadCovid;
                        atomicAdd(&AvailableBedsICU, 1);
                        printf("- Died in ICU, freed bed\n");
                    }
                }
                else {
                    population[personIdx].Swap = d_ICU;
                    printf("- Continuing ICU care\n");
                }
            }
        }
    }
}