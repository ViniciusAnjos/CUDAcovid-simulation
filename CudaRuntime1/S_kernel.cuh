__global__ void S_kernel(GPUPerson* population, unsigned int* rngStates, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * L) return;

    int i, j;
    to2D(idx, L, i, j);

    if (i > 0 && i <= L && j > 0 && j <= L) {
        int personIdx = to1D(i, j, L);
        if (population[personIdx].Health == d_S) {
            // Check for natural death
            if (population[personIdx].Days >= population[personIdx].AgeDeathDays) {
                population[personIdx].Swap = d_Dead;
            }
            else {
                // Check for infection through contacts
                ContactResult contacts = checkAllContacts(i, j, L, population, &rngStates[idx]);

                if (contacts.anyContact) {
                    double probInfection = calculateInfectionProbability(contacts.infectiousContacts);
                    double rn = generateRandom(&rngStates[idx]);

                    if (rn <= probInfection) {
                        population[personIdx].Checked = 1;
                        population[personIdx].Swap = d_E;
                        population[personIdx].TimeOnState = 0;

                        rn = generateRandom(&rngStates[idx]);
                        population[personIdx].StateTime =
                            rn * (d_MaxLatency - d_MinLatency) + d_MinLatency;
                    }
                    else {
                        population[personIdx].Swap = d_S;
                    }
                }
                else {
                    population[personIdx].Swap = d_S;
                }
            }
        }
    }
}