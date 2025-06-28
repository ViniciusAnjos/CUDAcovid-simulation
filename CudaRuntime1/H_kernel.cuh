// H_kernel.cuh - Corrected version using ProbHtoICU
__global__ void H_kernel(GPUPerson* population, unsigned int* rngStates, int L) {
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
                // Free hospital bed on natural death
                atomicAdd(&AvailableBeds, 1);
            }
            else {
                // Check if time in hospital is complete
                if (population[personIdx].TimeOnState >= population[personIdx].StateTime) {
                    // Get thread-specific RNG
                    unsigned int* myRNG = &rngStates[idx];

                    // Use age-specific recovery probability
                    int age = population[personIdx].AgeYears;
                    if (age > 120) age = 120; // Safety check

                    // Access the global device pointer for recovery probability
                    double recoveryProb = d_ProbRecoveryH[age];

                    // First check: Does patient recover?
                    double rn = generateRandom(myRNG);
                    if (rn < recoveryProb) {
                        // Patient recovers from hospital
                        population[personIdx].Swap = d_Recovered;

                        // Free up a hospital bed
                        atomicAdd(&AvailableBeds, 1);

                        // Increment new recovered counter
                        atomicAdd(&d_New_Recovered, 1);
                    }
                    else {
                        // Patient doesn't recover - try to get ICU (matches original logic)
                        int remainingICUBeds = atomicAdd(&AvailableBedsICU, -1);
                        if (remainingICUBeds > 0) {
                            // ICU bed available - move to ICU
                            population[personIdx].Swap = d_ICU;

                            // Free regular hospital bed, ICU bed already allocated
                            atomicAdd(&AvailableBeds, 1);

                            // Set time for ICU stay
                            rn = generateRandom(myRNG);
                            population[personIdx].StateTime = (int)(rn * (d_MaxICU - d_MinICU) + d_MinICU);

                            // Increment new ICU counter
                            atomicAdd(&d_New_ICU, 1);
                        }
                        else {
                            // No ICU bed available - patient dies
                            atomicAdd(&AvailableBedsICU, 1); // Restore ICU count
                            population[personIdx].Swap = d_DeadCovid;

                            // Free up hospital bed
                            atomicAdd(&AvailableBeds, 1);

                            // Increment new death counter
                            atomicAdd(&d_New_DeadCovid, 1);
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