__global__ void IP_kernel(GPUPerson* population, unsigned int* rngStates, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int i, j;
    to2D(idx, L, i, j);

    if (i > 0 && i <= L && j > 0 && j <= L) {
        int personIdx = to1D(i, j, L);
        if (population[personIdx].Health == d_IP) {
            population[personIdx].TimeOnState++;

            if (population[personIdx].Days >= population[personIdx].AgeDeathDays) {
                population[personIdx].Swap = d_Dead;
            }
            else {
                if (population[personIdx].TimeOnState >= population[personIdx].StateTime) {
                    double rn = generateRandom(&rngStates[idx]);

                    if (rn < d_ProbIPtoIA) {
                        population[personIdx].Swap = d_IA;
                        rn = generateRandom(&rngStates[idx]);
                        population[personIdx].StateTime =
                            rn * (d_MaxIA - d_MinIA) + d_MinIA;
                    }
                    else {
                        rn = generateRandom(&rngStates[idx]);

                        if (rn < d_ProbToBecomeISLight) {
                            population[personIdx].Swap = d_ISLight;
                            rn = generateRandom(&rngStates[idx]);
                            population[personIdx].StateTime =
                                rn * (d_MaxISLight - d_MinISLight) + d_MinISLight;
                        }
                        else if (rn < (d_ProbToBecomeISLight + d_ProbToBecomeISModerate)) {
                            population[personIdx].Swap = d_ISModerate;
                            rn = generateRandom(&rngStates[idx]);
                            population[personIdx].StateTime =
                                rn * (d_MaxISModerate - d_MinISModerate) + d_MinISModerate;
                        }
                        else {
                            population[personIdx].Swap = d_ISSevere;
                            rn = generateRandom(&rngStates[idx]);
                            population[personIdx].StateTime =
                                rn * (d_MaxISSevere - d_MinISSevere) + d_MinISSevere;
                        }
                    }
                }
                else {
                    spreadInfection_kernel(i, j, population, &rngStates[idx], L);
                    population[personIdx].Swap = d_IP;
                }
            }
        }
    }
}