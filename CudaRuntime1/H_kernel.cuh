__global__ void H_kernel(GPUPerson* population, unsigned int* rngStates, int L, int* hospitalBeds, int* icuBeds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * L) return;

    int i, j;
    to2D(idx, L, i, j);

    if (i > 0 && i <= L && j > 0 && j <= L) {
        int personIdx = to1D(i, j, L);

        // Print basic info for all cells
        printf("Cell (%d,%d): Health=%d, Days=%d, AgeDeathDays=%d\n",
            i, j, population[personIdx].Health,
            population[personIdx].Days,
            population[personIdx].AgeDeathDays);

        if (population[personIdx].Health == d_H) {
            printf("H patient (%d,%d): Before - Swap=%d\n", i, j, population[personIdx].Swap);

            // Increment time in current state
            population[personIdx].TimeOnState++;

            if (population[personIdx].Days >= population[personIdx].AgeDeathDays) {
                printf("Patient (%d,%d): Natural death triggered\n", i, j);
                population[personIdx].Swap = d_Dead;
                atomicAdd(hospitalBeds, 1);
            }
            else if (population[personIdx].TimeOnState >= population[personIdx].StateTime) {
                printf("Patient (%d,%d): Time complete, decision point\n", i, j);

                double rn = generateRandom(&rngStates[idx]);
                printf("Patient (%d,%d): RNG=%f\n", i, j, rn);

                // Get recovery probability for this age from device array
                int age = population[personIdx].AgeYears;
                if (age > 120) age = 120;  // Safety check

                // Debug for recovery probability
                printf("DEBUG: Patient (%d,%d): Age=%d, d_ProbRecoveryH address=%p\n",
                    i, j, age, d_ProbRecoveryH);

                double recoveryProb = d_ProbRecoveryH[age];

                printf("Patient (%d,%d): Age=%d, Recovery Prob=%f\n",
                    i, j, population[personIdx].AgeYears, recoveryProb);

                if (rn < recoveryProb) {
                    printf("Patient (%d,%d): Recovered\n", i, j);
                    population[personIdx].Swap = d_Recovered;
                    atomicAdd(hospitalBeds, 1);
                }
                else {
                    int currentICUBeds = atomicAdd(icuBeds, 0);
                    printf("Patient (%d,%d): Needs ICU, available=%d\n", i, j, currentICUBeds);

                    if (currentICUBeds > 0) {
                        printf("Patient (%d,%d): Moving to ICU\n", i, j);
                        population[personIdx].Swap = d_ICU;
                        atomicSub(icuBeds, 1);
                        atomicAdd(hospitalBeds, 1);

                        rn = generateRandom(&rngStates[idx]);
                        population[personIdx].StateTime = rn * (d_MaxICU - d_MinICU) + d_MinICU;
                    }
                    else {
                        printf("Patient (%d,%d): No ICU available, died\n", i, j);
                        population[personIdx].Swap = d_DeadCovid;
                        atomicAdd(hospitalBeds, 1);
                    }
                }
            }
            else {
                printf("Patient (%d,%d): Continuing hospital stay\n", i, j);
                population[personIdx].Swap = d_H;
            }

            printf("H patient (%d,%d): After - Swap=%d\n", i, j, population[personIdx].Swap);
        }
    }
}