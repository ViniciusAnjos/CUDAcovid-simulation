

// Initialize the entire population
// A more robust but simplified initialization kernel
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

        // Simplified demographic distribution - just use a uniform distribution
        double rn = generateRandom(myRNG);
        population[idx].AgeYears = (int)(rn * 90); // Ages 0-90
        population[idx].AgeDays = population[idx].AgeYears * 365;

        // Add some randomness to birth day
        rn = generateRandom(myRNG);
        population[idx].AgeDays += (int)(rn * 365);

        // Initialize state timers
        population[idx].TimeOnState = 0;
        population[idx].StateTime = 0;
        population[idx].Days = 0;

        // Simplified death age assignment - older than current age
        rn = generateRandom(myRNG);
        // Death age between current age+1 and 100
        int ageRange = 100 - population[idx].AgeYears;
        if (ageRange <= 0) ageRange = 1; // Ensure at least 1 year difference

        population[idx].AgeDeathYears = population[idx].AgeYears + 1 + (int)(rn * ageRange);
        if (population[idx].AgeDeathYears > 100) population[idx].AgeDeathYears = 100;

        population[idx].AgeDeathDays = population[idx].AgeDeathYears * 365;
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
    if (idx < 15) { // 11 different states
        stateCounts[idx] = 0;
        newCounts[idx] = 0;

        // Set initial S count to total population
        if (idx == d_S) {
            stateCounts[d_S] = N; // All start as susceptible
        }
    }
}
