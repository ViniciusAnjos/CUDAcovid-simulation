// Initialize the entire population with proper Brazilian age structure
__global__ void initPopulation_kernel(GPUPerson* population, unsigned int* rngStates, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (L + 2) * (L + 2)) return;

    int i, j;
    to2D(idx, L, i, j);

    // Skip boundary cells
    if (i > 0 && i <= L && j > 0 && j <= L) {
        // Get the RNG state for this thread
        unsigned int* myRNG = &rngStates[idx];

        // Initialize as Susceptible
        population[idx].Health = d_S;
        population[idx].Swap = d_S;
        population[idx].Isolation = d_IsolationNo;
        population[idx].Exponent = 0;
        population[idx].Checked = 0;

        // PROPER AGE ASSIGNMENT USING BRAZILIAN DEMOGRAPHIC DATA
        // This matches the original begin.h implementation
        int mute = 0;
        int k = 0;
        int MaximumAge, MinimumAge;

        double rn = generateRandom(myRNG);

        // Find the correct age group using cumulative probabilities
        if (rn <= d_SumProbBirthAge[k]) {
            mute = 1;
            MaximumAge = d_AgeMax[k];
            MinimumAge = d_AgeMin[k];
        }
        else {
            do {
                if (rn > d_SumProbBirthAge[k] && rn <= d_SumProbBirthAge[k + 1]) {
                    mute = 1;
                    MaximumAge = d_AgeMax[k + 1];
                    MinimumAge = d_AgeMin[k + 1];
                }
                else {
                    mute = 0;
                    k++;
                }
            } while (mute < 1 && k < 18); // Safety check to prevent infinite loop
        }

        // Assign age within the selected range
        rn = generateRandom(myRNG);
        population[idx].AgeYears = (int)(rn * (MaximumAge - MinimumAge) + MinimumAge);
        population[idx].AgeDays = population[idx].AgeYears * 365;

        // Add random birthday offset (like original)
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

    // Distribute IP infections
    for (int k = 0; k < IPini; k++) {
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

    // Save RNG state back
    rngStates[0] = localState;
}

// Initialize counters for tracking states
__global__ void initCounters_kernel(int* stateCounts, int* newCounts, int N) {
    int idx = threadIdx.x;

    // Initialize all counters to 0
    if (idx < 15) { // 15 different states
        stateCounts[idx] = 0;
        newCounts[idx] = 0;

        // Set initial S count to total population
        if (idx == d_S) {
            stateCounts[d_S] = N; // All start as susceptible
        }
    }
}