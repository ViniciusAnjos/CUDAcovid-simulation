
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
        d_S_Total = N;
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
    }
}

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

// FIX: Replaces dead person using the same rejection-sampling method as the
// original Update.h, drawing AgeDeathYears from ProbNaturalDeath distribution.
__device__ void replaceDeadPerson(GPUPerson* person, unsigned int* rngState,
                                   double* probNaturalDeath) {
    person->Health = d_S;
    person->Swap   = d_S;
    person->TimeOnState = 0;
    person->StateTime   = 0;
    person->Days        = 0;
    person->Exponent    = 0;
    person->Checked     = 0;
    person->Isolation   = d_IsolationNo;

    // Random current age (0-99), same as original
    double rn = generateRandom(rngState);
    person->AgeYears = (int)(rn * 100);
    person->AgeDays  = person->AgeYears * 365;

    // FIX: Rejection sampling for age of death using ProbNaturalDeath,
    // matching the original Update.h logic exactly.
    int mute = 0;
    do {
        rn = generateRandom(rngState);
        person->AgeDeathYears = (int)(rn * 100);

        rn = generateRandom(rngState);
        if (rn < probNaturalDeath[person->AgeDeathYears])
            mute = 1;
        else
            mute = 0;
    } while (mute < 1);

    person->AgeDeathDays = person->AgeDeathYears * 365;

    // FIX: Swap ages if death age < current age, matching original Update.h
    if (person->AgeDeathYears < person->AgeYears) {
        int tmp = person->AgeDeathYears;
        person->AgeDeathYears = person->AgeYears;
        person->AgeYears      = tmp;
        person->AgeDeathDays  = person->AgeDeathYears * 365;
    }
}

__global__ void update_kernel(GPUPerson* population, unsigned int* rngStates,
    int L, int day, double* probNaturalDeath) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (L + 2) * (L + 2)) return;

    int i, j;
    to2D(idx, L, i, j);

    if (i > 0 && i <= L && j > 0 && j <= L) {
        int personIdx = to1D(i, j, L);

        int oldState = population[personIdx].Health;
        int newState = population[personIdx].Swap;

        // Count state transitions (incidence)
        if (oldState != newState) {
            population[personIdx].Health = newState;

            if (newState == d_S)          atomicAdd(&d_New_S,          1);
            else if (newState == d_E)     atomicAdd(&d_New_E,          1);
            else if (newState == d_IP)    atomicAdd(&d_New_IP,         1);
            else if (newState == d_IA)    atomicAdd(&d_New_IA,         1);
            else if (newState == d_ISLight)    atomicAdd(&d_New_ISLight,    1);
            else if (newState == d_ISModerate) atomicAdd(&d_New_ISModerate, 1);
            else if (newState == d_ISSevere)   atomicAdd(&d_New_ISSevere,   1);
            else if (newState == d_H)     atomicAdd(&d_New_H,          1);
            else if (newState == d_ICU)   atomicAdd(&d_New_ICU,        1);
            else if (newState == d_Recovered) atomicAdd(&d_New_Recovered, 1);
            else if (newState == d_DeadCovid) {
                atomicAdd(&d_New_DeadCovid, 1);
                atomicAdd(&d_DeadCovid_Total, 1);
            }
            else if (newState == d_Dead) {
                atomicAdd(&d_New_Dead, 1);
                atomicAdd(&d_Dead_Total, 1);
            }
        }

        // FIX: Safety check from original Updatefunc() - if AgeYears reached
        // AgeDeathYears, force death (handles edge cases)
        if (population[personIdx].AgeYears >= population[personIdx].AgeDeathYears &&
            population[personIdx].Health != d_Dead &&
            population[personIdx].Health != d_DeadCovid) {
            population[personIdx].Health = d_Dead;
            atomicAdd(&d_New_Dead,  1);
            atomicAdd(&d_Dead_Total, 1);
        }

        // Replace dead persons with new susceptibles
        if (population[personIdx].Health == d_Dead ||
            population[personIdx].Health == d_DeadCovid) {
            replaceDeadPerson(&population[personIdx], &rngStates[idx], probNaturalDeath);
            atomicAdd(&d_New_S, 1);
        }

        // Count prevalence (current state after all changes)
        int finalState = population[personIdx].Health;
        if      (finalState == d_S)          atomicAdd(&d_S_Total,          1);
        else if (finalState == d_E)          atomicAdd(&d_E_Total,          1);
        else if (finalState == d_IP)         atomicAdd(&d_IP_Total,         1);
        else if (finalState == d_IA)         atomicAdd(&d_IA_Total,         1);
        else if (finalState == d_ISLight)    atomicAdd(&d_ISLight_Total,    1);
        else if (finalState == d_ISModerate) atomicAdd(&d_ISModerate_Total, 1);
        else if (finalState == d_ISSevere)   atomicAdd(&d_ISSevere_Total,   1);
        else if (finalState == d_H)          atomicAdd(&d_H_Total,          1);
        else if (finalState == d_ICU)        atomicAdd(&d_ICU_Total,        1);
        else if (finalState == d_Recovered)  atomicAdd(&d_Recovered_Total,  1);

        // FIX: Add AgeDays++ (was missing — matches original Updatefunc())
        population[personIdx].AgeDays++;

        // Days tracks days alive since birth/replacement (used in death checks)
        population[personIdx].Days++;

        // NOTE: TimeOnState is NOT incremented here.
        // Each state kernel (E, IP, IS, H, ICU) increments it before
        // checking the transition condition, matching the original .h files.
        population[personIdx].Exponent = 0;
        population[personIdx].Checked  = 0;
    }
}

__host__ void getCountersFromDevice(int* h_totals, int* h_new_cases) {
    cudaMemcpyFromSymbol(&h_totals[S],          d_S_Total,          sizeof(int));
    cudaMemcpyFromSymbol(&h_totals[E],          d_E_Total,          sizeof(int));
    cudaMemcpyFromSymbol(&h_totals[IP],         d_IP_Total,         sizeof(int));
    cudaMemcpyFromSymbol(&h_totals[IA],         d_IA_Total,         sizeof(int));
    cudaMemcpyFromSymbol(&h_totals[ISLight],    d_ISLight_Total,    sizeof(int));
    cudaMemcpyFromSymbol(&h_totals[ISModerate], d_ISModerate_Total, sizeof(int));
    cudaMemcpyFromSymbol(&h_totals[ISSevere],   d_ISSevere_Total,   sizeof(int));
    cudaMemcpyFromSymbol(&h_totals[H],          d_H_Total,          sizeof(int));
    cudaMemcpyFromSymbol(&h_totals[ICU],        d_ICU_Total,        sizeof(int));
    cudaMemcpyFromSymbol(&h_totals[Recovered],  d_Recovered_Total,  sizeof(int));
    cudaMemcpyFromSymbol(&h_totals[DeadCovid],  d_DeadCovid_Total,  sizeof(int));
    cudaMemcpyFromSymbol(&h_totals[Dead],       d_Dead_Total,       sizeof(int));

    cudaMemcpyFromSymbol(&h_new_cases[S],          d_New_S,          sizeof(int));
    cudaMemcpyFromSymbol(&h_new_cases[E],          d_New_E,          sizeof(int));
    cudaMemcpyFromSymbol(&h_new_cases[IP],         d_New_IP,         sizeof(int));
    cudaMemcpyFromSymbol(&h_new_cases[IA],         d_New_IA,         sizeof(int));
    cudaMemcpyFromSymbol(&h_new_cases[ISLight],    d_New_ISLight,    sizeof(int));
    cudaMemcpyFromSymbol(&h_new_cases[ISModerate], d_New_ISModerate, sizeof(int));
    cudaMemcpyFromSymbol(&h_new_cases[ISSevere],   d_New_ISSevere,   sizeof(int));
    cudaMemcpyFromSymbol(&h_new_cases[H],          d_New_H,          sizeof(int));
    cudaMemcpyFromSymbol(&h_new_cases[ICU],        d_New_ICU,        sizeof(int));
    cudaMemcpyFromSymbol(&h_new_cases[Recovered],  d_New_Recovered,  sizeof(int));
    cudaMemcpyFromSymbol(&h_new_cases[DeadCovid],  d_New_DeadCovid,  sizeof(int));
    cudaMemcpyFromSymbol(&h_new_cases[Dead],       d_New_Dead,       sizeof(int));
}
