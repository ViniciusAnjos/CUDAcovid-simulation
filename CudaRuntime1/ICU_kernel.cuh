__global__ void ICU_kernel(GPUPerson* population, unsigned int* rngStates, int L, int* hospitalBeds, int* icuBeds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (L + 2) * (L + 2)) return;

    int i, j;
    to2D(idx, L, i, j);

    if (i > 0 && i <= L && j > 0 && j <= L) {
        int personIdx = to1D(i, j, L);
        if (population[personIdx].Health == d_ICU) {
            // Increment time in current state
            population[personIdx].TimeOnState++;

            // Check for natural death
            if (population[personIdx].Days >= population[personIdx].AgeDeathDays) {
                population[personIdx].Swap = d_Dead;

                // Free up an ICU bed
                atomicAdd(icuBeds, 1);
            }
            else {
                // Check if time in ICU is complete
                if (population[personIdx].TimeOnState >= population[personIdx].StateTime) {
                    // Get thread-specific RNG
                    unsigned int* myRNG = &rngStates[idx];

                    // Use age-specific recovery probability from global device pointer
                    int age = population[personIdx].AgeYears;
                    if (age > 120) age = 120; // Safety check

                    // Access the global device pointer
                    double recoveryProb = d_ProbRecoveryICU[age];

                    // Determine outcome based on recovery probability
                    double rn = generateRandom(myRNG);
                    if (rn < recoveryProb) {
                        // Patient recovers from ICU
                        population[personIdx].Swap = d_Recovered;

                        // Free up an ICU bed
                        atomicAdd(icuBeds, 1);
                    }
                    else {
                        // Patient dies in ICU
                        population[personIdx].Swap = d_DeadCovid;

                        // Free up an ICU bed
                        atomicAdd(icuBeds, 1);
                    }
                }
                else {
                    // Continue ICU care
                    population[personIdx].Swap = d_ICU;
                }
            }
        }
    }
}