// test_update_kernel.cuh
// Unit tests for Update_kernel.cuh fixes.
//
// Tests verify:
//   1. TimeOnState is only incremented once per day (by state kernels, not update)
//   2. AgeDays is incremented each day
//   3. Death replacement uses rejection sampling via ProbNaturalDeath
//   4. AgeDeathYears >= AgeYears is enforced after replacement
//   5. State transition incidence counters are correct
//   6. Prevalence counters match actual population distribution

#pragma once
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

// ─── helpers ────────────────────────────────────────────────────────────────

static bool g_anyFailed = false;

#define TEST_CHECK(cond, name)                                             \
    do {                                                                   \
        if (!(cond)) {                                                     \
            printf("  FAIL  %s\n", name);                                  \
            g_anyFailed = true;                                            \
        } else {                                                           \
            printf("  PASS  %s\n", name);                                  \
        }                                                                  \
    } while (0)

// Copy a single __device__ int counter to host
static int readDeviceInt(const void* symbol) {
    int val;
    cudaMemcpyFromSymbol(&val, symbol, sizeof(int));
    return val;
}

// ─── shared setup ───────────────────────────────────────────────────────────

// Build minimal ProbNaturalDeath array on the host (same values as gpu_define.cuh)
static void buildProbNaturalDeath(double* p) {
    double vals[80] = {
        0.123582605937503, 0.008412659481117, 0.0053758830325,
        0.004071349564963, 0.003328785199437, 0.002851304993677,
        0.002527966451363, 0.002310496945863, 0.002178859462886,
        0.002129830660268, 0.002173822944408, 0.002335864060389,
        0.002659754298987, 0.00321570391484,  0.004112609101089,
        0.006950173136317, 0.008664660102385, 0.010186902957357,
        0.011378739645498, 0.01229421773494,  0.013200885964887,
        0.014099674432299, 0.014711957238925, 0.014968691990926,
        0.014967172185325, 0.014845031894506, 0.014767082029733,
        0.01480839152999,  0.015055101638628, 0.015468505190105,
        0.015944306819536, 0.016421043709446, 0.016942117099386,
        0.017501078390588, 0.018119607730054, 0.018840382831522,
        0.01968584321183,  0.020648697917729, 0.02174025422673,
        0.022977134791163, 0.024351327962323, 0.025902332950567,
        0.02769484883125,  0.029763807831847, 0.032092257416785,
        0.03464417578323,  0.037387253883753, 0.040325116908918,
        0.043450465802503, 0.046783769250641, 0.050379305647723,
        0.054250708597469, 0.058370165793835, 0.062743356857589,
        0.067405733722802, 0.072471872691788, 0.077940570749002,
        0.083722432777535, 0.089812939592727, 0.096320545551754,
        0.103374758030823, 0.111153832260191, 0.119799418763301,
        0.129458206950975, 0.140179867078005, 0.151763459015659,
        0.164404993288517, 0.178637575194583, 0.194752162647952,
        0.212709619046675, 0.232085117364568, 0.25292358002509,
        0.27584171899225,  0.301132600796412, 0.328832323976937,
        0.358580946581084, 0.390547125436947, 0.425516392727306,
        0.463966442471834, 0.506035659978245
    };
    for (int i = 0; i < 80; i++) p[i] = vals[i];
    for (int i = 80; i <= 120; i++) p[i] = 1.0;
}

// ─── Test 1: AgeDays is incremented by update_kernel ────────────────────────

static void test_AgeDays_increment() {
    printf("\n[Test 1] AgeDays incremented by update_kernel\n");

    const int L = 4;
    const int gridSize = (L + 2) * (L + 2);

    GPUPerson* h_pop = new GPUPerson[gridSize]();
    // Place one S person with known AgeDays at (1,1)
    int idx11 = to1D(1, 1, L);
    h_pop[idx11].Health        = S;
    h_pop[idx11].Swap          = S;
    h_pop[idx11].AgeDays       = 10000;
    h_pop[idx11].Days          = 10000;
    h_pop[idx11].AgeYears      = 27;
    h_pop[idx11].AgeDeathYears = 80;
    h_pop[idx11].AgeDeathDays  = 80 * 365;

    GPUPerson* d_pop;
    cudaMalloc(&d_pop, gridSize * sizeof(GPUPerson));
    cudaMemcpy(d_pop, h_pop, gridSize * sizeof(GPUPerson), cudaMemcpyHostToDevice);

    unsigned int* d_rng;
    cudaMalloc(&d_rng, gridSize * sizeof(unsigned int));
    // init RNG to non-zero
    cudaMemset(d_rng, 1, gridSize * sizeof(unsigned int));

    double h_prob[121];
    buildProbNaturalDeath(h_prob);
    double* d_prob;
    cudaMalloc(&d_prob, 121 * sizeof(double));
    cudaMemcpy(d_prob, h_prob, 121 * sizeof(double), cudaMemcpyHostToDevice);

    resetCounters_kernel<<<1,1>>>();
    resetNewCounters_kernel<<<1,1>>>();
    cudaDeviceSynchronize();

    update_kernel<<<1, gridSize>>>(d_pop, d_rng, L, 1, d_prob);
    cudaDeviceSynchronize();

    cudaMemcpy(h_pop, d_pop, gridSize * sizeof(GPUPerson), cudaMemcpyDeviceToHost);

    TEST_CHECK(h_pop[idx11].AgeDays == 10001,
               "AgeDays incremented from 10000 to 10001");
    TEST_CHECK(h_pop[idx11].Days == 10001,
               "Days incremented from 10000 to 10001");

    delete[] h_pop;
    cudaFree(d_pop); cudaFree(d_rng); cudaFree(d_prob);
}

// ─── Test 2: TimeOnState is NOT incremented by update_kernel ────────────────
// State kernels (E, IP, IS, H, ICU) own TimeOnState increments.
// update_kernel must not double-count it.

static void test_TimeOnState_not_double_incremented() {
    printf("\n[Test 2] update_kernel does NOT increment TimeOnState\n");

    const int L = 4;
    const int gridSize = (L + 2) * (L + 2);

    GPUPerson* h_pop = new GPUPerson[gridSize]();
    int idx = to1D(1, 1, L);
    // Person in E state, stays E (TimeOnState < StateTime)
    h_pop[idx].Health        = E;
    h_pop[idx].Swap          = E;   // no transition this day
    h_pop[idx].TimeOnState   = 3;
    h_pop[idx].StateTime     = 10;
    h_pop[idx].Days          = 0;
    h_pop[idx].AgeYears      = 30;
    h_pop[idx].AgeDeathYears = 80;
    h_pop[idx].AgeDeathDays  = 80 * 365;

    GPUPerson* d_pop;
    cudaMalloc(&d_pop, gridSize * sizeof(GPUPerson));
    cudaMemcpy(d_pop, h_pop, gridSize * sizeof(GPUPerson), cudaMemcpyHostToDevice);

    unsigned int* d_rng;
    cudaMalloc(&d_rng, gridSize * sizeof(unsigned int));
    cudaMemset(d_rng, 1, gridSize * sizeof(unsigned int));

    double h_prob[121]; buildProbNaturalDeath(h_prob);
    double* d_prob;
    cudaMalloc(&d_prob, 121 * sizeof(double));
    cudaMemcpy(d_prob, h_prob, 121 * sizeof(double), cudaMemcpyHostToDevice);

    resetCounters_kernel<<<1,1>>>();
    resetNewCounters_kernel<<<1,1>>>();
    cudaDeviceSynchronize();

    // Run update_kernel only (no state kernel) — TimeOnState must stay at 3
    update_kernel<<<1, gridSize>>>(d_pop, d_rng, L, 1, d_prob);
    cudaDeviceSynchronize();

    cudaMemcpy(h_pop, d_pop, gridSize * sizeof(GPUPerson), cudaMemcpyDeviceToHost);

    TEST_CHECK(h_pop[idx].TimeOnState == 3,
               "TimeOnState unchanged by update_kernel (stays 3)");
    TEST_CHECK(h_pop[idx].Health == E,
               "Health remains E when Swap == E");

    delete[] h_pop;
    cudaFree(d_pop); cudaFree(d_rng); cudaFree(d_prob);
}

// ─── Test 3: Dead person is replaced with new susceptible ───────────────────

static void test_Dead_replacement() {
    printf("\n[Test 3] Dead person replaced by new S with valid ages\n");

    const int L = 4;
    const int gridSize = (L + 2) * (L + 2);

    GPUPerson* h_pop = new GPUPerson[gridSize]();
    int idx = to1D(2, 2, L);
    h_pop[idx].Health        = Dead;
    h_pop[idx].Swap          = Dead;
    h_pop[idx].AgeYears      = 70;
    h_pop[idx].AgeDeathYears = 71;
    h_pop[idx].AgeDeathDays  = 71 * 365;
    h_pop[idx].Days          = 71 * 365 + 1;   // just died

    GPUPerson* d_pop;
    cudaMalloc(&d_pop, gridSize * sizeof(GPUPerson));
    cudaMemcpy(d_pop, h_pop, gridSize * sizeof(GPUPerson), cudaMemcpyHostToDevice);

    unsigned int* d_rng;
    cudaMalloc(&d_rng, gridSize * sizeof(unsigned int));
    // Seed with something non-trivial
    unsigned int h_rng[gridSize];
    for (int i = 0; i < gridSize; i++) h_rng[i] = 893221891u * (i + 1);
    cudaMemcpy(d_rng, h_rng, gridSize * sizeof(unsigned int), cudaMemcpyHostToDevice);

    double h_prob[121]; buildProbNaturalDeath(h_prob);
    double* d_prob;
    cudaMalloc(&d_prob, 121 * sizeof(double));
    cudaMemcpy(d_prob, h_prob, 121 * sizeof(double), cudaMemcpyHostToDevice);

    resetCounters_kernel<<<1,1>>>();
    resetNewCounters_kernel<<<1,1>>>();
    cudaDeviceSynchronize();

    update_kernel<<<1, gridSize>>>(d_pop, d_rng, L, 1, d_prob);
    cudaDeviceSynchronize();

    cudaMemcpy(h_pop, d_pop, gridSize * sizeof(GPUPerson), cudaMemcpyDeviceToHost);

    TEST_CHECK(h_pop[idx].Health == S,
               "Replaced person is Susceptible");
    TEST_CHECK(h_pop[idx].Days == 1,   // 0 on replacement + Days++ at end
               "Days reset to 0 and then incremented to 1");
    TEST_CHECK(h_pop[idx].AgeDeathYears >= h_pop[idx].AgeYears,
               "AgeDeathYears >= AgeYears after replacement");
    TEST_CHECK(h_pop[idx].AgeDeathYears >= 0 && h_pop[idx].AgeDeathYears <= 100,
               "AgeDeathYears in valid range [0,100]");

    delete[] h_pop;
    cudaFree(d_pop); cudaFree(d_rng); cudaFree(d_prob);
}

// ─── Test 4: AgeDeathYears >= AgeYears invariant holds over many replacements

static void test_AgeDeathYears_invariant() {
    printf("\n[Test 4] AgeDeathYears >= AgeYears holds for all replaced persons\n");

    const int L = 8;
    const int gridSize = (L + 2) * (L + 2);

    GPUPerson* h_pop = new GPUPerson[gridSize]();
    // Make all inner cells dead so all get replaced
    for (int i = 1; i <= L; i++)
        for (int j = 1; j <= L; j++) {
            int k = to1D(i, j, L);
            h_pop[k].Health  = Dead;
            h_pop[k].Swap    = Dead;
            h_pop[k].Days    = 99999;
        }

    GPUPerson* d_pop;
    cudaMalloc(&d_pop, gridSize * sizeof(GPUPerson));
    cudaMemcpy(d_pop, h_pop, gridSize * sizeof(GPUPerson), cudaMemcpyHostToDevice);

    unsigned int* d_rng;
    cudaMalloc(&d_rng, gridSize * sizeof(unsigned int));
    unsigned int h_rng[gridSize];
    for (int i = 0; i < gridSize; i++) h_rng[i] = 893221891u * (i + 7);
    cudaMemcpy(d_rng, h_rng, gridSize * sizeof(unsigned int), cudaMemcpyHostToDevice);

    double h_prob[121]; buildProbNaturalDeath(h_prob);
    double* d_prob;
    cudaMalloc(&d_prob, 121 * sizeof(double));
    cudaMemcpy(d_prob, h_prob, 121 * sizeof(double), cudaMemcpyHostToDevice);

    resetCounters_kernel<<<1,1>>>();
    resetNewCounters_kernel<<<1,1>>>();
    cudaDeviceSynchronize();

    update_kernel<<<1, gridSize>>>(d_pop, d_rng, L, 1, d_prob);
    cudaDeviceSynchronize();

    cudaMemcpy(h_pop, d_pop, gridSize * sizeof(GPUPerson), cudaMemcpyDeviceToHost);

    bool allOk = true;
    for (int i = 1; i <= L; i++)
        for (int j = 1; j <= L; j++) {
            int k = to1D(i, j, L);
            if (h_pop[k].AgeDeathYears < h_pop[k].AgeYears) {
                allOk = false;
                printf("    FAIL at (%d,%d): AgeYears=%d AgeDeathYears=%d\n",
                       i, j, h_pop[k].AgeYears, h_pop[k].AgeDeathYears);
            }
        }

    TEST_CHECK(allOk,
               "AgeDeathYears >= AgeYears for all 64 replaced persons");

    delete[] h_pop;
    cudaFree(d_pop); cudaFree(d_rng); cudaFree(d_prob);
}

// ─── Test 5: Incidence counters track transitions correctly ─────────────────

static void test_Incidence_counters() {
    printf("\n[Test 5] Incidence counters reflect state transitions\n");

    const int L = 4;
    const int gridSize = (L + 2) * (L + 2);

    GPUPerson* h_pop = new GPUPerson[gridSize]();

    // Person 1: E → IP transition
    int idx1 = to1D(1, 1, L);
    h_pop[idx1].Health        = E;
    h_pop[idx1].Swap          = IP;  // transition set by E_kernel
    h_pop[idx1].Days          = 0;
    h_pop[idx1].AgeYears      = 25;
    h_pop[idx1].AgeDeathYears = 80;
    h_pop[idx1].AgeDeathDays  = 80 * 365;

    // Person 2: S stays S (no transition)
    int idx2 = to1D(1, 2, L);
    h_pop[idx2].Health        = S;
    h_pop[idx2].Swap          = S;
    h_pop[idx2].Days          = 0;
    h_pop[idx2].AgeYears      = 30;
    h_pop[idx2].AgeDeathYears = 80;
    h_pop[idx2].AgeDeathDays  = 80 * 365;

    // Person 3: IP → ISLight transition
    int idx3 = to1D(2, 1, L);
    h_pop[idx3].Health        = IP;
    h_pop[idx3].Swap          = ISLight;
    h_pop[idx3].Days          = 0;
    h_pop[idx3].AgeYears      = 40;
    h_pop[idx3].AgeDeathYears = 80;
    h_pop[idx3].AgeDeathDays  = 80 * 365;

    GPUPerson* d_pop;
    cudaMalloc(&d_pop, gridSize * sizeof(GPUPerson));
    cudaMemcpy(d_pop, h_pop, gridSize * sizeof(GPUPerson), cudaMemcpyHostToDevice);

    unsigned int* d_rng;
    cudaMalloc(&d_rng, gridSize * sizeof(unsigned int));
    cudaMemset(d_rng, 1, gridSize * sizeof(unsigned int));

    double h_prob[121]; buildProbNaturalDeath(h_prob);
    double* d_prob;
    cudaMalloc(&d_prob, 121 * sizeof(double));
    cudaMemcpy(d_prob, h_prob, 121 * sizeof(double), cudaMemcpyHostToDevice);

    resetCounters_kernel<<<1,1>>>();
    resetNewCounters_kernel<<<1,1>>>();
    cudaDeviceSynchronize();

    update_kernel<<<1, gridSize>>>(d_pop, d_rng, L, 1, d_prob);
    cudaDeviceSynchronize();

    int h_totals[15] = {0}, h_new[15] = {0};
    getCountersFromDevice(h_totals, h_new);

    TEST_CHECK(h_new[IP]      == 1, "New_IP == 1 (E->IP transition counted)");
    TEST_CHECK(h_new[ISLight] == 1, "New_ISLight == 1 (IP->ISLight counted)");
    TEST_CHECK(h_new[E]       == 0, "New_E == 0 (no new E exposures)");

    delete[] h_pop;
    cudaFree(d_pop); cudaFree(d_rng); cudaFree(d_prob);
}

// ─── Test 6: Prevalence counters match actual population ────────────────────

static void test_Prevalence_counters() {
    printf("\n[Test 6] Prevalence counters match population after update\n");

    const int L = 4;
    const int gridSize = (L + 2) * (L + 2);

    GPUPerson* h_pop = new GPUPerson[gridSize]();
    // Place known states in inner cells
    int nS = 0, nE = 0, nH = 0;
    int inner = 0;
    for (int i = 1; i <= L; i++)
        for (int j = 1; j <= L; j++) {
            int k = to1D(i, j, L);
            h_pop[k].AgeYears      = 30;
            h_pop[k].AgeDeathYears = 80;
            h_pop[k].AgeDeathDays  = 80 * 365;
            h_pop[k].Days          = 0;

            int state;
            if      (inner % 3 == 0) { state = S;  nS++; }
            else if (inner % 3 == 1) { state = E;  nE++; }
            else                      { state = H;  nH++; }
            h_pop[k].Health = state;
            h_pop[k].Swap   = state; // no transitions this step
            inner++;
        }

    GPUPerson* d_pop;
    cudaMalloc(&d_pop, gridSize * sizeof(GPUPerson));
    cudaMemcpy(d_pop, h_pop, gridSize * sizeof(GPUPerson), cudaMemcpyHostToDevice);

    unsigned int* d_rng;
    cudaMalloc(&d_rng, gridSize * sizeof(unsigned int));
    cudaMemset(d_rng, 1, gridSize * sizeof(unsigned int));

    double h_prob[121]; buildProbNaturalDeath(h_prob);
    double* d_prob;
    cudaMalloc(&d_prob, 121 * sizeof(double));
    cudaMemcpy(d_prob, h_prob, 121 * sizeof(double), cudaMemcpyHostToDevice);

    resetCounters_kernel<<<1,1>>>();
    resetNewCounters_kernel<<<1,1>>>();
    cudaDeviceSynchronize();

    update_kernel<<<1, gridSize>>>(d_pop, d_rng, L, 1, d_prob);
    cudaDeviceSynchronize();

    int h_totals[15] = {0}, h_new[15] = {0};
    getCountersFromDevice(h_totals, h_new);

    TEST_CHECK(h_totals[S] == nS, "S prevalence counter matches expected S count");
    TEST_CHECK(h_totals[E] == nE, "E prevalence counter matches expected E count");
    TEST_CHECK(h_totals[H] == nH, "H prevalence counter matches expected H count");

    delete[] h_pop;
    cudaFree(d_pop); cudaFree(d_rng); cudaFree(d_prob);
}

// ─── Test runner ─────────────────────────────────────────────────────────────

inline int runUpdateKernelTests() {
    printf("=== Update_kernel Unit Tests ===\n");

    test_AgeDays_increment();
    test_TimeOnState_not_double_incremented();
    test_Dead_replacement();
    test_AgeDeathYears_invariant();
    test_Incidence_counters();
    test_Prevalence_counters();

    printf("\n");
    if (g_anyFailed) {
        printf("Result: SOME TESTS FAILED\n");
        return 1;
    }
    printf("Result: ALL TESTS PASSED\n");
    return 0;
}
