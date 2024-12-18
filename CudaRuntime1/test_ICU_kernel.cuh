void test_ICU_state() {
    const int L = 5;
    const int gridSize = (L + 2) * (L + 2);

    // Allocate and initialize test population
    GPUPerson* h_population = new GPUPerson[gridSize];
    GPUPerson* d_population;
    cudaMalloc(&d_population, gridSize * sizeof(GPUPerson));

    // Initialize all as Susceptible
    printf("\nInitializing population...\n");
    for (int i = 0; i < gridSize; i++) {
        h_population[i].Health = S;
        h_population[i].Swap = S;
        h_population[i].TimeOnState = 0;
        h_population[i].StateTime = 0;
        h_population[i].Days = 0;
        h_population[i].AgeYears = 40;  // Middle-aged by default
        h_population[i].AgeDeathDays = 365 * 80;
    }

    // Set up test cases
    printf("\nSetting up test cases:\n");

    // Case 1: Young person in ICU (better recovery chance)
    printf("\nCase 1 - Young ICU case (2,2):\n");
    h_population[to1D(2, 2, L)].Health = ICU;
    h_population[to1D(2, 2, L)].StateTime = 5;
    h_population[to1D(2, 2, L)].TimeOnState = 5;
    h_population[to1D(2, 2, L)].AgeYears = 25;
    printf("- Age: %d years\n", h_population[to1D(2, 2, L)].AgeYears);

    // Case 2: Elderly person in ICU (lower recovery chance)
    printf("\nCase 2 - Elderly ICU case (3,3):\n");
    h_population[to1D(3, 3, L)].Health = ICU;
    h_population[to1D(3, 3, L)].StateTime = 5;
    h_population[to1D(3, 3, L)].TimeOnState = 5;
    h_population[to1D(3, 3, L)].AgeYears = 75;
    printf("- Age: %d years\n", h_population[to1D(3, 3, L)].AgeYears);

    // Case 3: Natural death case
    printf("\nCase 3 - Natural Death Case (4,4):\n");
    h_population[to1D(4, 4, L)].Health = ICU;
    h_population[to1D(4, 4, L)].Days = 365 * 85;  // 85 years
    printf("- Days: %d (over age limit)\n", h_population[to1D(4, 4, L)].Days);

    // Set initial ICU beds
    AvailableBedsICU = 10;
    printf("\nInitial Available ICU Beds: %d\n", AvailableBedsICU);

    // Print initial grid
    printf("\nInitial Grid State:\n");
    printf("S: Susceptible, I: ICU\n");
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            printf("%c ", h_population[to1D(i, j, L)].Health == ICU ? 'I' : 'S');
        }
        printf("\n");
    }

    // Copy to device and run kernel
    cudaMemcpy(d_population, h_population, gridSize * sizeof(GPUPerson),
        cudaMemcpyHostToDevice);

    setupRNG(gridSize);

    int blockSize = 256;
    int numBlocks = (gridSize + blockSize - 1) / blockSize;
    printf("\nLaunching ICU_kernel with %d blocks, %d threads per block\n",
        numBlocks, blockSize);

    ICU_kernel << <numBlocks, blockSize >> > (d_population, d_rngStates, L);

    // Get results
    cudaMemcpy(h_population, d_population, gridSize * sizeof(GPUPerson),
        cudaMemcpyDeviceToHost);

    // Print final state
    printf("\nFinal Grid State:\n");
    printf("S: Susceptible, I: ICU, R: Recovered, D: Dead\n");
    for (int i = 1; i <= L; i++) {
        for (int j = 1; j <= L; j++) {
            char state = 'S';
            int idx = to1D(i, j, L);
            switch (h_population[idx].Swap) {
            case ICU: state = 'I'; break;
            case Recovered: state = 'R'; break;
            case Dead: case DeadCovid: state = 'D'; break;
            }
            printf("%c ", state);
        }
        printf("\n");
    }

    // Print detailed case results
    printf("\nDetailed Final State for Test Cases:\n");

    printf("\nCase 1 (2,2) - Young Patient Final State:\n");
    int idx = to1D(2, 2, L);
    printf("- Health: %d\n", h_population[idx].Health);
    printf("- Swap: %d\n", h_population[idx].Swap);
    printf("- TimeOnState: %d\n", h_population[idx].TimeOnState);
    printf("- Age: %d\n", h_population[idx].AgeYears);

    printf("\nCase 2 (3,3) - Elderly Patient Final State:\n");
    idx = to1D(3, 3, L);
    printf("- Health: %d\n", h_population[idx].Health);
    printf("- Swap: %d\n", h_population[idx].Swap);
    printf("- TimeOnState: %d\n", h_population[idx].TimeOnState);
    printf("- Age: %d\n", h_population[idx].AgeYears);

    printf("\nCase 3 (4,4) - Natural Death Case Final State:\n");
    idx = to1D(4, 4, L);
    printf("- Health: %d\n", h_population[idx].Health);
    printf("- Swap: %d\n", h_population[idx].Swap);
    printf("- Days: %d\n", h_population[idx].Days);

    printf("\nFinal Available ICU Beds: %d\n", AvailableBedsICU);

    // Cleanup
    delete[] h_population;
    cudaFree(d_population);
    cleanupRNG();
}

int main() {
    int city = ROC;
    setupCityParameters(city);
    setupGPUConstants();

    test_ICU_state();

    cleanupGPUConstants();
    return 0;
}