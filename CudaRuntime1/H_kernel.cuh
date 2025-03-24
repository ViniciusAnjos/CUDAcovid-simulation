__global__ void H_kernel(GPUPerson* population, unsigned int* rngStates, int L, int* hospitalBeds, int* icuBeds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (L + 2) * (L + 2)) return;

    int i, j;
    to2D(idx, L, i, j);

    if (i > 0 && i <= L && j > 0 && j <= L) {
        int personIdx = to1D(i, j, L);
        if (population[personIdx].Health == d_H) {
            // Increment time in current state
            population[personIdx].TimeOnState++;

            // Check for natural death
            if (population[personIdx].Days >= population[personIdx].AgeDeathDays) {
                population[personIdx].Swap = d_Dead;
            }
            else {
                // Check if time in hospital is complete
                if (population[personIdx].TimeOnState >= population[personIdx].StateTime) {
                    // Get thread-specific RNG
                    unsigned int* myRNG = &rngStates[idx];

                    // Use age-specific recovery probability from global device pointer
                    int age = population[personIdx].AgeYears;
                    if (age > 120) age = 120; // Safety check

                    // Access the global device pointer
                    double recoveryProb = d_ProbRecoveryH[age];

                    // Determine outcome based on recovery probability
                    double rn = generateRandom(myRNG);
                    if (rn < recoveryProb) {
                        // Patient recovers
                        population[personIdx].Swap = d_Recovered;

                        // Free up a hospital bed
                        atomicAdd(hospitalBeds, 1);
                    }
                    else {
                        // Patient needs ICU
                        if (atomicAdd(icuBeds, 0) > 0) {  // Check ICU availability
                            // Move to ICU if bed is available
                            population[personIdx].Swap = d_ICU;

                            // Allocate ICU bed and free regular hospital bed
                            atomicAdd(icuBeds, -1);
                            atomicAdd(hospitalBeds, 1);

                            // Set time for ICU stay
                            rn = generateRandom(myRNG);
                            population[personIdx].StateTime = (int)(rn * (d_MaxICU - d_MinICU) + d_MinICU);
                        }
                        else {
                            // No ICU bed available, patient dies
                            population[personIdx].Swap = d_DeadCovid;

                            // Free up a hospital bed
                            atomicAdd(hospitalBeds, 1);
                        }
                    }
                }
                else {
                    // Continue hospital stay
                    population[personIdx].Swap = d_H;
                }
            }
        }
    }
}