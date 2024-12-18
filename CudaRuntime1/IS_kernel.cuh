__global__ void IS_kernel(GPUPerson* population, unsigned int* rngStates, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int i, j;
    to2D(idx, L, i, j);
    
    if (i > 0 && i <= L && j > 0 && j <= L) {
        int personIdx = to1D(i, j, L);
        
        // Check if person is in any IS statejjj
        int health = population[personIdx].Health;
        if(health == d_IA || health == d_ISLight || 
           health == d_ISModerate || health == d_ISSevere) {
            
            population[personIdx].TimeOnState++;
            
            if(population[personIdx].Days >= population[personIdx].AgeDeathDays) {
                population[personIdx].Swap = d_Dead;
            }
            else {
                // Handle IA case
                if(health == d_IA) {
                    if(population[personIdx].TimeOnState >= population[personIdx].StateTime) {
                        population[personIdx].Swap = d_Recovered;
                    }
                    else {
                        spreadInfection_kernel(i, j, population, &rngStates[idx], L);
                        population[personIdx].Swap = d_IA;
                    }
                }
                // Handle ISLight case
                else if(health == d_ISLight) {
                    if(population[personIdx].TimeOnState >= population[personIdx].StateTime) {
                        double rn = generateRandom(&rngStates[idx]);
                        if(rn < d_ProbISLightToISModerate) {
                            population[personIdx].Swap = d_ISModerate;
                            rn = generateRandom(&rngStates[idx]);
                            population[personIdx].StateTime = 
                                rn * (d_MaxISModerate - d_MinISModerate) + d_MinISModerate;
                        }
                        else {
                            population[personIdx].Swap = d_Recovered;
                        }
                    }
                    else {
                        spreadInfection_kernel(i, j, population, &rngStates[idx], L);
                        population[personIdx].Swap = d_ISLight;
                    }
                }
                // Handle ISModerate case
                else if(health == d_ISModerate) {
                    if(population[personIdx].TimeOnState >= population[personIdx].StateTime) {
                        double rn = generateRandom(&rngStates[idx]);
                        if(rn < d_ProbRecoveryModerate[population[personIdx].AgeYears]) {
                            population[personIdx].Swap = d_Recovered;
                        }
                        else {
                            population[personIdx].Swap = d_ISSevere;
                            rn = generateRandom(&rngStates[idx]);
                            population[personIdx].StateTime = 
                                rn * (d_MaxISSevere - d_MinISSevere) + d_MinISSevere;
                        }
                    }
                    else {
                        population[personIdx].Swap = d_ISModerate;
                    }
                }
                // Handle ISSevere case
                else if(health == d_ISSevere) {
                    if(population[personIdx].TimeOnState >= population[personIdx].StateTime) {
                        double rn = generateRandom(&rngStates[idx]);
                        if(rn < d_ProbRecoverySevere[population[personIdx].AgeYears]) {
                            population[personIdx].Swap = d_Recovered;
                        }
                        else {
                            if(AvailableBeds > 0) {
                                atomicAdd(&AvailableBeds, -1);
                                population[personIdx].Swap = d_H;
                                rn = generateRandom(&rngStates[idx]);
                                population[personIdx].StateTime = 
                                    rn * (d_MaxH - d_MinH) + d_MinH;
                            }
                            else {
                                population[personIdx].Swap = d_DeadCovid;
                            }
                        }
                    }
                    else {
                        population[personIdx].Swap = d_ISSevere;
                    }
                }
            }
        }
    }
}