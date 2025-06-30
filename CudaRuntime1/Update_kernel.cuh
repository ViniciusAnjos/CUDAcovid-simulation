

// Global counters that need to be allocated
__device__ int d_S_Total;
__device__ int d_E_Total;
__device__ int d_IP_Total;
__device__ int d_IA_Total;
__device__ int d_ISLight_Total;
__device__ int d_ISModerate_Total;
__device__ int d_ISSevere_Total;
__device__ int d_H_Total;
__device__ int d_ICU_Total;
__device__ int d_Recovered_Total;
__device__ int d_DeadCovid_Total;
__device__ int d_Dead_Total;

// New case counters
__device__ int d_New_S;
__device__ int d_New_E;
__device__ int d_New_IP;
__device__ int d_New_IA;
__device__ int d_New_ISLight;
__device__ int d_New_ISModerate;
__device__ int d_New_ISSevere;
__device__ int d_New_H;
__device__ int d_New_ICU;
__device__ int d_New_Recovered;
__device__ int d_New_DeadCovid;
__device__ int d_New_Dead;

__global__ void initSimulationCounters_kernel(int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Initialize prevalence counters
        d_S_Total = N;  // Start with full susceptible population
        d_E_Total = 0;
        d_IP_Total = 0;
        d_IA_Total = 0;
        d_ISLight_Total = 0;
        d_ISModerate_Total = 0;
        d_ISSevere_Total = 0;
        d_H_Total = 0;
        d_ICU_Total = 0;
        d_Recovered_Total = 0;

        // Initialize cumulative death counters to ZERO at simulation start
        d_DeadCovid_Total = 0;  // Start fresh for this simulation
        d_Dead_Total = 0;       // Start fresh for this simulation

        // Initialize incidence counters
        d_New_S = 0;
        d_New_E = 0;
        d_New_IP = 0;
        d_New_IA = 0;
        d_New_ISLight = 0;
        d_New_ISModerate = 0;
        d_New_ISSevere = 0;
        d_New_H = 0;
        d_New_ICU = 0;
        d_New_Recovered = 0;
        d_New_DeadCovid = 0;
        d_New_Dead = 0;
    }
}
// CORRECTED: Kernel to reset counters at the beginning of each simulation
__global__ void resetCounters_kernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_S_Total = 0;
        d_E_Total = 0;
        d_IP_Total = 0;
        d_IA_Total = 0;
        d_ISLight_Total = 0;
        d_ISModerate_Total = 0;
        d_ISSevere_Total = 0;
        d_H_Total = 0;
        d_ICU_Total = 0;
        d_Recovered_Total = 0;
   // FIX: Reset natural deaths for each simulation
    }
}

// Kernel to reset new case counters 
__global__ void resetNewCounters_kernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_New_S = 0;
        d_New_E = 0;
        d_New_IP = 0;
        d_New_IA = 0;
        d_New_ISLight = 0;
        d_New_ISModerate = 0;
        d_New_ISSevere = 0;
        d_New_H = 0;
        d_New_ICU = 0;
        d_New_Recovered = 0;
        d_New_DeadCovid = 0;
        d_New_Dead = 0;
    }
}

// CORRECTED: Main update kernel with proper incidence counting and death handling
__global__ void update_kernel(GPUPerson* population, unsigned int* rngStates,
    int L, int day, double* probNaturalDeath) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (L + 2) * (L + 2)) return;

    int i, j;
    to2D(idx, L, i, j);

    if (i > 0 && i <= L && j > 0 && j <= L) {
        int personIdx = to1D(i, j, L);

        // STEP 1: Check for state transitions and COUNT THEM (INCIDENCE)
        int oldState = population[personIdx].Health;
        int newState = population[personIdx].Swap;

        // Only process if there's actually a state change
        if (oldState != newState) {
            // Update the person's health state
            population[personIdx].Health = newState;
            population[personIdx].TimeOnState = 0;  // Reset time in new state

            // COUNT THE TRANSITION (INCIDENCE) - Critical for proper tracking!
            if (newState == S) {
                atomicAdd(&d_New_S, 1);
            }
            else if (newState == E) {
                atomicAdd(&d_New_E, 1);
            }
            else if (newState == IP) {
                atomicAdd(&d_New_IP, 1);
            }
            else if (newState == IA) {
                atomicAdd(&d_New_IA, 1);
            }
            else if (newState == ISLight) {
                atomicAdd(&d_New_ISLight, 1);
            }
            else if (newState == ISModerate) {
                atomicAdd(&d_New_ISModerate, 1);
            }
            else if (newState == ISSevere) {
                atomicAdd(&d_New_ISSevere, 1);
            }
            else if (newState == H) {
                atomicAdd(&d_New_H, 1);
            }
            else if (newState == ICU) {
                atomicAdd(&d_New_ICU, 1);
            }
            else if (newState == Recovered) {
                atomicAdd(&d_New_Recovered, 1);
            }
            else if (newState == DeadCovid) {
                atomicAdd(&d_New_DeadCovid, 1);
                atomicAdd(&d_DeadCovid_Total, 1);  // Accumulate total COVID deaths
            }
            else if (newState == Dead) {
                atomicAdd(&d_New_Dead, 1);
                atomicAdd(&d_Dead_Total, 1);      // Accumulate total natural deaths
            }
        }

        // STEP 2: Handle death replacement (birth of new susceptible)
        if (population[personIdx].Health == Dead || population[personIdx].Health == DeadCovid) {
            // Replace dead person with new susceptible baby
            population[personIdx].Health = S;
            population[personIdx].Swap = S;
            population[personIdx].TimeOnState = 0;
            population[personIdx].Days = 0;

            // Assign new age at death based on age structure
            double rn = generateRandom(&rngStates[idx]);

            // Simple age assignment - could be made more sophisticated
            if (rn < 0.3) {
                population[personIdx].AgeYears = (int)(generateRandom(&rngStates[idx]) * 20);  // 0-19 years
            }
            else if (rn < 0.6) {
                population[personIdx].AgeYears = 20 + (int)(generateRandom(&rngStates[idx]) * 40);  // 20-59 years
            }
            else {
                population[personIdx].AgeYears = 60 + (int)(generateRandom(&rngStates[idx]) * 30);  // 60-89 years
            }

            population[personIdx].AgeDeathDays = population[personIdx].AgeYears * 365 +
                (int)(generateRandom(&rngStates[idx]) * 365);

            // Count as new birth
            atomicAdd(&d_New_S, 1);
        }

        // STEP 3: Count current states (PREVALENCE)
        if (population[personIdx].Health == S) {
            atomicAdd(&d_S_Total, 1);
        }
        else if (population[personIdx].Health == E) {
            atomicAdd(&d_E_Total, 1);
        }
        else if (population[personIdx].Health == IP) {
            atomicAdd(&d_IP_Total, 1);
        }
        else if (population[personIdx].Health == IA) {
            atomicAdd(&d_IA_Total, 1);
        }
        else if (population[personIdx].Health == ISLight) {
            atomicAdd(&d_ISLight_Total, 1);
        }
        else if (population[personIdx].Health == ISModerate) {
            atomicAdd(&d_ISModerate_Total, 1);
        }
        else if (population[personIdx].Health == ISSevere) {
            atomicAdd(&d_ISSevere_Total, 1);
        }
        else if (population[personIdx].Health == H) {
            atomicAdd(&d_H_Total, 1);
        }
        else if (population[personIdx].Health == ICU) {
            atomicAdd(&d_ICU_Total, 1);
        }
        else if (population[personIdx].Health == Recovered) {
            atomicAdd(&d_Recovered_Total, 1);
        }
        // Note: Don't count dead people in prevalence since they're immediately replaced

        // STEP 4: Update person's daily age and time in state
        population[personIdx].Days++;
        population[personIdx].TimeOnState++;
    }
}

// Note: initCounters_kernel is already defined in gpu_begin.cuh
// Note: distributeInitialInfections_kernel is already defined in gpu_begin.cuh

// Host function to get counter values from device
__host__ void getCountersFromDevice(int* h_totals, int* h_new_cases) {
    // Copy prevalence counters
    cudaMemcpyFromSymbol(&h_totals[S], d_S_Total, sizeof(int));
    cudaMemcpyFromSymbol(&h_totals[E], d_E_Total, sizeof(int));
    cudaMemcpyFromSymbol(&h_totals[IP], d_IP_Total, sizeof(int));
    cudaMemcpyFromSymbol(&h_totals[IA], d_IA_Total, sizeof(int));
    cudaMemcpyFromSymbol(&h_totals[ISLight], d_ISLight_Total, sizeof(int));
    cudaMemcpyFromSymbol(&h_totals[ISModerate], d_ISModerate_Total, sizeof(int));
    cudaMemcpyFromSymbol(&h_totals[ISSevere], d_ISSevere_Total, sizeof(int));
    cudaMemcpyFromSymbol(&h_totals[H], d_H_Total, sizeof(int));
    cudaMemcpyFromSymbol(&h_totals[ICU], d_ICU_Total, sizeof(int));
    cudaMemcpyFromSymbol(&h_totals[Recovered], d_Recovered_Total, sizeof(int));
    cudaMemcpyFromSymbol(&h_totals[DeadCovid], d_DeadCovid_Total, sizeof(int));
    cudaMemcpyFromSymbol(&h_totals[Dead], d_Dead_Total, sizeof(int));

    // Copy incidence counters
    cudaMemcpyFromSymbol(&h_new_cases[S], d_New_S, sizeof(int));
    cudaMemcpyFromSymbol(&h_new_cases[E], d_New_E, sizeof(int));
    cudaMemcpyFromSymbol(&h_new_cases[IP], d_New_IP, sizeof(int));
    cudaMemcpyFromSymbol(&h_new_cases[IA], d_New_IA, sizeof(int));
    cudaMemcpyFromSymbol(&h_new_cases[ISLight], d_New_ISLight, sizeof(int));
    cudaMemcpyFromSymbol(&h_new_cases[ISModerate], d_New_ISModerate, sizeof(int));
    cudaMemcpyFromSymbol(&h_new_cases[ISSevere], d_New_ISSevere, sizeof(int));
    cudaMemcpyFromSymbol(&h_new_cases[H], d_New_H, sizeof(int));
    cudaMemcpyFromSymbol(&h_new_cases[ICU], d_New_ICU, sizeof(int));
    cudaMemcpyFromSymbol(&h_new_cases[Recovered], d_New_Recovered, sizeof(int));
    cudaMemcpyFromSymbol(&h_new_cases[DeadCovid], d_New_DeadCovid, sizeof(int));
    cudaMemcpyFromSymbol(&h_new_cases[Dead], d_New_Dead, sizeof(int));
}


