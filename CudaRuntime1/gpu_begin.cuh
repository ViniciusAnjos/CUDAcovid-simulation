


// Initialize population with age distribution and health states
__global__ void initPopulation_kernel(GPUPerson* population, unsigned int* rngStates, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = (L + 2) * (L + 2);

    if (idx < gridSize) {
        unsigned int* myRNG = &rngStates[idx];
        int i, j;
        to2D(idx, L, i, j);

        // Initialize all population as susceptible
        population[idx].Health = d_S;
        population[idx].Swap = d_S;
        population[idx].PatientZeroID = 0;  // Initialize everyone as non-patient-zero

        // Initialize other fields
        population[idx].Gender = 0;
        population[idx].Isolation = d_IsolationNo;
        population[idx].Exponent = 0;
        population[idx].Checked = 0;

        // Only initialize inner grid cells (not boundaries)
        if (i > 0 && i <= L && j > 0 && j <= L) {
            // Age assignment using birth probability distribution
            int mute = 0;
            double rn;
            do {
                rn = generateRandom(myRNG);
                population[idx].AgeYears = (int)(rn * 100);

                rn = generateRandom(myRNG);
                if (rn < d_ProbBirthAge[population[idx].AgeYears]) {
                    mute = 1; // accept this age
                }
                else {
                    mute = 0; // reject, try again
                }
            } while (mute < 1);

            // Random age in days within the year
            rn = generateRandom(myRNG);
            population[idx].AgeDays += (int)(rn * 365);

            // Initialize state timers
            population[idx].TimeOnState = 0;
            population[idx].StateTime = 0;
            population[idx].Days = 0;

            // PROPER DEATH AGE ASSIGNMENT using rejection sampling (like original)
            // Define age of natural death using probability distribution
            mute = 0;
            do {
                rn = generateRandom(myRNG);
                population[idx].AgeDeathYears = (int)(rn * 100);

                rn = generateRandom(myRNG);
                if (rn < d_ProbNaturalDeath[population[idx].AgeDeathYears]) {
                    mute = 1; // accept this death age
                }
                else {
                    mute = 0; // reject, try again
                }
            } while (mute < 1);

            population[idx].AgeDeathDays = population[idx].AgeDeathYears * 365;

            // Safety check: ensure death age is after current age (like original)
            if (population[idx].AgeDeathYears < population[idx].AgeYears) {
                // Swap them if death age is somehow less than current age
                int temp = population[idx].AgeDeathYears;
                population[idx].AgeYears = population[idx].AgeDeathYears;
                population[idx].AgeDeathYears = temp;
                population[idx].AgeDeathDays = population[idx].AgeDeathYears * 365;
            }
        }
    }
}

// Distribute initial infections
__global__ void distributeInitialInfections_kernel(GPUPerson* population,
    unsigned int* rngStates,
    int* stateCounts,
    int* newCounts,
    int L,
    int Eini,
    int IPini,
    int IAini,
    int ISLightini,
    int ISModerateini,
    int ISSevereini) {
    // Only thread 0 does this work
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // Use first RNG state
    unsigned int localState = rngStates[0];

    // Distribute E infections
    for (int k = 0; k < Eini; k++) {
        int i, j, personIdx;

        // Find a susceptible person
        do {
            double rn = generateRandom(&localState);
            i = (int)(rn * L) + 1;

            rn = generateRandom(&localState);
            j = (int)(rn * L) + 1;

            personIdx = to1D(i, j, L);
        } while (population[personIdx].Health != d_S);

        // Change to Exposed
        population[personIdx].Health = d_E;
        population[personIdx].Swap = d_E;

        double rn = generateRandom(&localState);
        population[personIdx].StateTime = (int)(rn * (d_MaxLatency - d_MinLatency) + d_MinLatency);

        population[personIdx].Isolation = d_IsolationNo;

        // Update counters
        atomicAdd(&stateCounts[d_S], -1);
        atomicAdd(&stateCounts[d_E], 1);
        atomicAdd(&newCounts[d_E], 1);
    }

    // Distribute IP infections (Patient Zero mode modification)
#ifdef PATIENT_ZERO_ONLY_MODE
    for (int k = 0; k < 1; k++) {  // Force exactly 1 IP case
#else
    for (int k = 0; k < IPini; k++) {  // Normal mode
#endif
        int i, j, personIdx;

        // Find a susceptible person
        do {
            double rn = generateRandom(&localState);
            i = (int)(rn * L) + 1;

            rn = generateRandom(&localState);
            j = (int)(rn * L) + 1;

            personIdx = to1D(i, j, L);
        } while (population[personIdx].Health != d_S);

        // Change to IP
        population[personIdx].Health = d_IP;
        population[personIdx].Swap = d_IP;

        double rn = generateRandom(&localState);
        population[personIdx].StateTime = (int)(rn * (d_MaxIP - d_MinIP) + d_MinIP);

        population[personIdx].Isolation = d_IsolationNo;

#ifdef PATIENT_ZERO_ONLY_MODE
        if (k == 0) {
            population[personIdx].PatientZeroID = 1;  // Mark as patient zero
        }
#else
        population[personIdx].PatientZeroID = 1;  // In normal mode, all can infect
#endif

        // Update counters
        atomicAdd(&stateCounts[d_S], -1);
        atomicAdd(&stateCounts[d_IP], 1);
        atomicAdd(&newCounts[d_IP], 1);
    }

    // Distribute IA infections
    for (int k = 0; k < IAini; k++) {
        int i, j, personIdx;

        // Find a susceptible person
        do {
            double rn = generateRandom(&localState);
            i = (int)(rn * L) + 1;

            rn = generateRandom(&localState);
            j = (int)(rn * L) + 1;

            personIdx = to1D(i, j, L);
        } while (population[personIdx].Health != d_S);

        // Change to IA
        population[personIdx].Health = d_IA;
        population[personIdx].Swap = d_IA;

        double rn = generateRandom(&localState);
        population[personIdx].StateTime = (int)(rn * (d_MaxIA - d_MinIA) + d_MinIA);

        population[personIdx].Isolation = d_IsolationNo;

        // Update counters
        atomicAdd(&stateCounts[d_S], -1);
        atomicAdd(&stateCounts[d_IA], 1);
        atomicAdd(&newCounts[d_IA], 1);
    }

    // Distribute ISLight infections
    for (int k = 0; k < ISLightini; k++) {
        int i, j, personIdx;

        // Find a susceptible person
        do {
            double rn = generateRandom(&localState);
            i = (int)(rn * L) + 1;

            rn = generateRandom(&localState);
            j = (int)(rn * L) + 1;

            personIdx = to1D(i, j, L);
        } while (population[personIdx].Health != d_S);

        // Change to ISLight
        population[personIdx].Health = d_ISLight;
        population[personIdx].Swap = d_ISLight;

        double rn = generateRandom(&localState);
        population[personIdx].StateTime = (int)(rn * (d_MaxISLight - d_MinISLight) + d_MinISLight);

        population[personIdx].Isolation = d_IsolationNo;

        // Update counters
        atomicAdd(&stateCounts[d_S], -1);
        atomicAdd(&stateCounts[d_ISLight], 1);
        atomicAdd(&newCounts[d_ISLight], 1);
    }

    // Distribute ISModerate infections
    for (int k = 0; k < ISModerateini; k++) {
        int i, j, personIdx;

        // Find a susceptible person
        do {
            double rn = generateRandom(&localState);
            i = (int)(rn * L) + 1;

            rn = generateRandom(&localState);
            j = (int)(rn * L) + 1;

            personIdx = to1D(i, j, L);
        } while (population[personIdx].Health != d_S);

        // Change to ISModerate
        population[personIdx].Health = d_ISModerate;
        population[personIdx].Swap = d_ISModerate;

        double rn = generateRandom(&localState);
        population[personIdx].StateTime = (int)(rn * (d_MaxISModerate - d_MinISModerate) + d_MinISModerate);

        population[personIdx].Isolation = d_IsolationNo;

        // Update counters
        atomicAdd(&stateCounts[d_S], -1);
        atomicAdd(&stateCounts[d_ISModerate], 1);
        atomicAdd(&newCounts[d_ISModerate], 1);
    }

    // Distribute ISSevere infections
    for (int k = 0; k < ISSevereini; k++) {
        int i, j, personIdx;

        // Find a susceptible person
        do {
            double rn = generateRandom(&localState);
            i = (int)(rn * L) + 1;

            rn = generateRandom(&localState);
            j = (int)(rn * L) + 1;

            personIdx = to1D(i, j, L);
        } while (population[personIdx].Health != d_S);

        // Change to ISSevere
        population[personIdx].Health = d_ISSevere;
        population[personIdx].Swap = d_ISSevere;

        double rn = generateRandom(&localState);
        population[personIdx].StateTime = (int)(rn * (d_MaxISSevere - d_MinISSevere) + d_MinISSevere);

        population[personIdx].Isolation = d_IsolationNo;

        // Update counters
        atomicAdd(&stateCounts[d_S], -1);
        atomicAdd(&stateCounts[d_ISSevere], 1);
        atomicAdd(&newCounts[d_ISSevere], 1);
    }

    // Save updated RNG state
    rngStates[0] = localState;
    }

// Initialize counters kernel
__global__ void initCounters_kernel(int* stateCounts, int* newCounts, int N) {
    int idx = threadIdx.x;

    if (idx < 15) {
        if (idx == d_S) {
            stateCounts[idx] = N;  // All start as susceptible
        }
        else {
            stateCounts[idx] = 0;
        }
        newCounts[idx] = 0;
    }
}

