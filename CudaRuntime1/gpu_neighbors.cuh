

// Structure to hold contact checking results
struct ContactResult {
    int infectiousContacts;    // Total number of infectious contacts found
    bool anyContact;           // Whether any contact was made
};

// Function to check if a person is in an infectious state
__host__ __device__ inline bool isInfectious(int health) {
    return (health == d_IP || health == d_ISLight ||
        health == d_ISModerate || health == d_ISSevere ||
        health == d_IA || health == d_H || health == d_ICU);
}

// Function to handle local neighbor checking
__device__ ContactResult checkLocalContacts(int i, int j, int L, GPUPerson* population) {
    ContactResult result = { 0, false };

    // Get neighbor indices based on density
    int neighborIndices[8];
    int numNeighbors;
    getNeighborIndices(i, j, L, d_Density, neighborIndices, numNeighbors);

    // Check each neighbor
    for (int n = 0; n < numNeighbors; n++) {
        if (isInfectious(population[neighborIndices[n]].Health)) {
            result.infectiousContacts++;
            result.anyContact = true;
        }
    }

    return result;
}

// Function to handle random contacts
__device__ ContactResult checkRandomContacts(int i, int j, int L,
    GPUPerson* population,
    unsigned int* rngState) {
    ContactResult result = { 0, false };

    // Skip if person is isolated
    if (population[to1D(i, j, L)].Isolation == d_IsolationYes) {
        return result;
    }

    // Determine number of random contacts
    double rn = generateRandom(rngState);
    int randomContacts = (int)(rn * (d_MaxRandomContacts - d_MinRandomContacts) +
        d_MinRandomContacts);

    // Process each random contact
    for (int contact = 0; contact < randomContacts; contact++) {
        // Generate random position (avoiding self)
        int randI, randJ;
        do {
            rn = generateRandom(rngState);
            randI = (int)(rn * L) + 1;

            rn = generateRandom(rngState);
            randJ = (int)(rn * L) + 1;
        } while (randI == i && randJ == j);

        // Check if random contact is infectious
        int randIdx = to1D(randI, randJ, L);
        if (isInfectious(population[randIdx].Health)) {
            result.infectiousContacts++;
            result.anyContact = true;
        }
    }

    return result;
}

// Main function to check all contacts (both local and random)
__device__ ContactResult checkAllContacts(int i, int j, int L,
    GPUPerson* population,
    unsigned int* rngState) {
    // Check local contacts first
    ContactResult localResult = checkLocalContacts(i, j, L, population);

    // Then check random contacts
    ContactResult randomResult = checkRandomContacts(i, j, L, population, rngState);

    // Combine results
    ContactResult totalResult = {
        localResult.infectiousContacts + randomResult.infectiousContacts,
        localResult.anyContact || randomResult.anyContact
    };

    return totalResult;
}

__device__ void spreadInfection_kernel(int i, int j, GPUPerson* population,
    unsigned int* rngState, int L) {

    double rn = generateRandom(rngState);
    int randomContacts = (int)(rn * (d_MaxRandomContacts - d_MinRandomContacts) +
        d_MinRandomContacts);


    for (int contact = 0; contact < randomContacts; contact++) {
        int Randomi, Randomj;
        do {
            rn = generateRandom(rngState);
            Randomi = (int)(rn * L) + 1;

            rn = generateRandom(rngState);
            Randomj = (int)(rn * L) + 1;
        } while (Randomi == i && Randomj == j);

        int randomIdx = to1D(Randomi, Randomj, L);

        if (population[randomIdx].Health == d_S) {

            int oldval = atomicAdd((int*)&population[randomIdx].Exponent, 1);

            if (population[randomIdx].Checked == 0 && oldval == 0) {
                double ProbContagion = 1.0 - pow(1.0 - d_Beta, (double)(oldval + 1));

                rn = generateRandom(rngState);


                if (rn <= ProbContagion) {
                    population[randomIdx].Checked = 1;
                    population[randomIdx].Swap = d_E;
                    population[randomIdx].TimeOnState = 0;

                    rn = generateRandom(rngState);
                    population[randomIdx].StateTime = rn * (d_MaxLatency - d_MinLatency) + d_MinLatency;


                }
            }
        }
    }
}


// Calculate infection probability based on number of contacts
__device__ double calculateInfectionProbability(int infectiousContacts) {
    if (infectiousContacts == 0) return 0.0;
    return 1.0 - pow(1.0 - d_Beta, (double)infectiousContacts);
}



