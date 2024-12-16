
__global__ void E_kernel(GPUPerson* population, unsigned int* rngStates, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * L) return;

    int i, j;
    to2D(idx, L, i, j);

    if (i > 0 && i <= L && j > 0 && j <= L) {
        int personIdx = to1D(i, j, L);
        if (population[personIdx].Health == d_E) {
            population[personIdx].TimeOnState++;

            // Check for natural death
            if (population[personIdx].Days >= population[personIdx].AgeDeathDays) {
                population[personIdx].Swap = d_Dead;
                // Note: New_Dead counter will be handled in update kernel
            }
            else {  // did not die (natural death)
                if (population[personIdx].TimeOnState >= population[personIdx].StateTime) {
                    // time in E state is over
                    population[personIdx].Swap = d_IP;

                    // Set time in state IP
                    double rn = generateRandom(&rngStates[idx]);
                    population[personIdx].StateTime = rn * (d_MaxIP - d_MinIP) + d_MinIP;

                    // Note: New_IP counter will be handled in update kernel
                }
                else {
                    population[personIdx].Swap = d_E;
                }
            }
        }
    }
}