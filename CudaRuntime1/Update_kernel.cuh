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

// Kernel to reset counters at the beginning of update
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
        d_DeadCovid_Total = 0;
        d_Dead_Total = 0;
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

// Main update kernel
__global__ void update_kernel(GPUPerson* population, unsigned int* rngStates,
    int L, int day, double* probNaturalDeath) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (L + 2) * (L + 2)) return;

    int i, j;
    to2D(idx, L, i, j);

    if (i > 0 && i <= L && j > 0 && j <= L) {
        int personIdx = to1D(i, j, L);

        // STEP 1: Update lattice from Swap values (this is critical!)
        population[personIdx].Health = population[personIdx].Swap;

        // STEP 2: Reset exposure tracking
        population[personIdx].Exponent = 0;
        population[personIdx].Checked = 0;

        // STEP 3: Age the person
        population[personIdx].AgeDays++;
        population[personIdx].Days++;

        // STEP 4: Check for natural death by age (from original Update.h line 63)
        if (population[personIdx].AgeYears >= population[personIdx].AgeDeathYears) {
            population[personIdx].Health = d_Dead;
        }

        // STEP 5: Handle death (natural or COVID) - replace with new susceptible
        // This is the KEY LOGIC from original Update.h lines 34-85
        if (population[personIdx].Health == d_Dead ||
            population[personIdx].Health == d_DeadCovid) {

            // Count the death BEFORE replacement
            if (population[personIdx].Health == d_Dead) {
                atomicAdd(&d_New_Dead, 1);
                atomicAdd(&d_Dead_Total, 1);
                //printf("Natural death counted at (%d,%d)\n", i, j);
            }
            else if (population[personIdx].Health == d_DeadCovid) {
                atomicAdd(&d_New_DeadCovid, 1);
                atomicAdd(&d_DeadCovid_Total, 1);
                //printf("COVID death counted at (%d,%d)\n", i, j);
            }

            // Replace with new susceptible person (from original Update.h)
            population[personIdx].Health = d_S;
            population[personIdx].Swap = d_S;

            // Assign new random age using RNG (from original Update.h lines 53-54)
            unsigned int* myRNG = &rngStates[idx];
            double rn = generateRandom(myRNG);
            population[personIdx].AgeYears = (int)(rn * 100);
            population[personIdx].AgeDays = population[personIdx].AgeYears * 365;
            population[personIdx].Days = 0;

            // Reset state timers
            population[personIdx].TimeOnState = 0;
            population[personIdx].StateTime = 0;

            // Assign new death age using probability table (from original Update.h lines 59-68)
            bool assigned = false;
            int attempts = 0;
            do {
                rn = generateRandom(myRNG);
                int candidateDeathAge = (int)(rn * 100);

                rn = generateRandom(myRNG);
                if (rn < probNaturalDeath[candidateDeathAge]) {
                    population[personIdx].AgeDeathYears = candidateDeathAge;
                    population[personIdx].AgeDeathDays = candidateDeathAge * 365;
                    assigned = true;
                }
                attempts++;
            } while (!assigned && attempts < 100);

            // Fallback if probabilistic assignment fails
            if (!assigned) {
                population[personIdx].AgeDeathYears = population[personIdx].AgeYears + 20;
                population[personIdx].AgeDeathDays = population[personIdx].AgeDeathYears * 365;
            }

            // Ensure death age is after current age (from original Update.h lines 70-76)
            if (population[personIdx].AgeDeathYears < population[personIdx].AgeYears) {
                int temp = population[personIdx].AgeDeathYears;
                population[personIdx].AgeYears = temp;
                population[personIdx].AgeDeathYears = population[personIdx].AgeYears + 10;
                population[personIdx].AgeDeathDays = population[personIdx].AgeDeathYears * 365;
            }

            // Count new susceptible
            atomicAdd(&d_New_S, 1);
            atomicAdd(&d_S_Total, 1);
        }
        // STEP 6: Count current states for living people
        else{ 
        int health = population[personIdx].Health;
        if (health == d_S) {
            atomicAdd(&d_S_Total, 1);
        }
        else if (health == d_E) {
            atomicAdd(&d_E_Total, 1);
        }
        else if (health == d_IP) {
            atomicAdd(&d_IP_Total, 1);
        }
        else if (health == d_IA) {
            atomicAdd(&d_IA_Total, 1);
        }
        else if (health == d_ISLight) {
            atomicAdd(&d_ISLight_Total, 1);
        }
        else if (health == d_ISModerate) {
            atomicAdd(&d_ISModerate_Total, 1);
        }
        else if (health == d_ISSevere) {
            atomicAdd(&d_ISSevere_Total, 1);
        }
        else if (health == d_H) {
            atomicAdd(&d_H_Total, 1);
        }
        else if (health == d_ICU) {
            atomicAdd(&d_ICU_Total, 1);
        }
        else if (health == d_Recovered) {
            atomicAdd(&d_Recovered_Total, 1);
        }

        }
    }
}

// Also need this kernel to properly reset counters at the start of each day
__global__ void resetTotalCounters_kernel() {
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
        // Note: Don't reset d_DeadCovid_Total and d_Dead_Total as they accumulate
    }
}

// Host function to get counter values
__host__ void getCountersFromDevice(int* h_totals, int* h_new_cases) {
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