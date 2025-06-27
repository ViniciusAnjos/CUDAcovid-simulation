// ICU_kernel.cuh - Corrected version using global symbols
__global__ void ICU_kernel(GPUPerson* population, unsigned int* rngStates, int L) {
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

                // Free up an ICU bed on natural death
                atomicAdd(&AvailableBedsICU, 1);
            }
            else {
                // Check if time in ICU is complete
                if (population[personIdx].TimeOnState >= population[personIdx].StateTime) {
                    // Get thread-specific RNG
                    unsigned int* myRNG = &rngStates[idx];

                    // Use age-specific recovery probability
                    int age = population[personIdx].AgeYears;
                    if (age > 120) age = 120; // Safety check

                    // Access the global device pointer for ICU recovery probability
                    double recoveryProb = d_ProbRecoveryICU[age];

                    // Determine outcome based on recovery probability
                    double rn = generateRandom(myRNG);
                    if (rn < recoveryProb) {
                        // Patient recovers from ICU
                        population[personIdx].Swap = d_Recovered;

                        // Free up an ICU bed
                        atomicAdd(&AvailableBedsICU, 1);

                        // Increment new recovered counter
                        atomicAdd(&d_New_Recovered, 1);
                    }
                    else {
                        // Patient dies in ICU
                        population[personIdx].Swap = d_DeadCovid;

                        // Free up an ICU bed
                        atomicAdd(&AvailableBedsICU, 1);

                        // Increment new COVID death counter
                        atomicAdd(&d_New_DeadCovid, 1);
                    }
                }
                else {
                    // Continue ICU stay
                    population[personIdx].Swap = d_ICU;
                }
            }
        }
    }
}