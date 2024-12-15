#ifndef GPU_CONSTANTS_CUH
#define GPU_CONSTANTS_CUH

#include "define.h"
#include "cities.h"

//cities(ROC);




// Error checking function
void checkCudaError(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(error));
        exit(-1);
    }
}

// Disease States Constants
__constant__ int d_S;
__constant__ int d_E;
__constant__ int d_IP;
__constant__ int d_IA;
__constant__ int d_IS;
__constant__ int d_H;
__constant__ int d_ICU;
__constant__ int d_Recovered;
__constant__ int d_DeadCovid;
__constant__ int d_Dead;
__constant__ int d_ISLight;
__constant__ int d_ISModerate;
__constant__ int d_ISSevere;

// Simulation Constants
__constant__ int d_L;
__constant__ int d_N;
__constant__ int d_DAYS;
__constant__ int d_MAXSIM;

// Initial values
__constant__ int d_Eini;
__constant__ int d_IPini;
__constant__ int d_IAini;
__constant__ int d_ISLightini;
__constant__ int d_ISModerateini;
__constant__ int d_ISSevereini;
__constant__ int d_Hini;
__constant__ int d_ICUini;
__constant__ int d_Recoveredini;

// Disease Parameters
__constant__ double d_Beta;
__constant__ double d_ProbIPtoIA;
__constant__ double d_ProbHtoICU;
__constant__ double d_ProbToBecomeISLight;
__constant__ double d_ProbToBecomeISModerate;
__constant__ double d_ProbToBecomeISSevere;
__constant__ double d_ProbISLightToISModerate;

// Time Period Constants
__constant__ double d_MinLatency, d_MaxLatency;
__constant__ double d_MinIA, d_MaxIA;
__constant__ double d_MinIP, d_MaxIP;
__constant__ double d_MinISLight, d_MaxISLight;
__constant__ double d_MinISModerate, d_MaxISModerate;
__constant__ double d_MinISSevere, d_MaxISSevere;
__constant__ double d_MinH, d_MaxH;
__constant__ double d_MinICU, d_MaxICU;

// Health System Constants
__constant__ double d_AverageOcupationRateBeds;
__constant__ double d_AverageOcupationRateBedsICU;

// City Parameters
__constant__ double d_BEDSPOP;
__constant__ double d_ICUPOP;
__constant__ double d_MaxRandomContacts;
__constant__ double d_MinRandomContacts;
__constant__ int d_Density;
__constant__ int d_HIGH;
__constant__ int d_LOW;
__constant__ int d_SP;    // São Paulo
__constant__ int d_ROC;   // Rocinha
__constant__ int d_BRA;   // Brasília
__constant__ int d_MAN;   // Manaus
__constant__ int d_C5;
__constant__ int d_C6;
__constant__ int d_C7;


// Constants for isolation
__constant__ int d_IsolationYes;
__constant__ int d_IsolationNo;
__constant__ int d_ON;
__constant__ int d_OFF;

// Global arrays (pointers)
__device__ double* d_ProbNaturalDeath;
__device__ double* d_ProbRecoveryModerate;
__device__ double* d_ProbRecoverySevere;
__device__ double* d_ProbRecoveryH;
__device__ double* d_ProbRecoveryICU;

__device__ double* d_ProbBirthAge;
__device__ double* d_SumProbBirthAge;
__device__ int* d_AgeMin;
__device__ int* d_AgeMax;

__host__ void setupCityParameters(int city) {
    // These values will be copied to GPU constants
    double bedspop = 0.0;
    double icupop = 0.0;
    double maxContacts = 0.0;
    double minContacts = 0.0;
    int density = LOW;

    // Use the same switch from cities.h
    switch (city) {
    case SP:  // São Paulo
        density = HIGH;
        bedspop = 0.00247452;
        icupop = 0.00043782;
        maxContacts = 2.5;
        minContacts = 1.5;
        break;

    case ROC:  // Rocinha
        density = HIGH;
        bedspop = 0.00055111;
        icupop = 0.00014592;
        maxContacts = 119.5;
        minContacts = 1.5;
        break;

    case BRA:  // Brasília
        density = LOW;
        bedspop = 0.00260879;
        icupop = 0.00040114;
        maxContacts = 2.5;
        minContacts = 1.5;
        break;

    case MAN:  // Manaus
        density = LOW;
        bedspop = 0.00187124;
        icupop = 0.00027858;
        maxContacts = 2.5;
        minContacts = 1.5;
        break;

    case C5:
    case C6:
    case C7:
        density = HIGH;
        break;
    }
}



// Function to build recovery probability arrays
__host__ void buildArrays(
    double* h_ProbNaturalDeath,
    double* h_ProbRecoveryModerate,
    double* h_ProbRecoverySevere,
    double* h_ProbRecoveryH,
    double* h_ProbRecoveryICU,
    double* h_ProbBirthAge,
    double* h_SumProbBirthAge,
    int* h_AgeMin,
    int* h_AgeMax) {

    // Natural Death Probabilities
    h_ProbNaturalDeath[0] = 0.123582605937503;
    h_ProbNaturalDeath[1] = 0.008412659481117;
    h_ProbNaturalDeath[2] = 0.0053758830325;
    h_ProbNaturalDeath[3] = 0.004071349564963;
    h_ProbNaturalDeath[4] = 0.003328785199437;
    h_ProbNaturalDeath[5] = 0.002851304993677;
    h_ProbNaturalDeath[6] = 0.002527966451363;
    h_ProbNaturalDeath[7] = 0.002310496945863;
    h_ProbNaturalDeath[8] = 0.002178859462886;
    h_ProbNaturalDeath[9] = 0.002129830660268;
    h_ProbNaturalDeath[10] = 0.002173822944408;
    h_ProbNaturalDeath[11] = 0.002335864060389;
    h_ProbNaturalDeath[12] = 0.002659754298987;
    h_ProbNaturalDeath[13] = 0.00321570391484;
    h_ProbNaturalDeath[14] = 0.004112609101089;
    h_ProbNaturalDeath[15] = 0.006950173136317;
    h_ProbNaturalDeath[16] = 0.008664660102385;
    h_ProbNaturalDeath[17] = 0.010186902957357;
    h_ProbNaturalDeath[18] = 0.011378739645498;
    h_ProbNaturalDeath[19] = 0.01229421773494;
    h_ProbNaturalDeath[20] = 0.013200885964887;
    h_ProbNaturalDeath[21] = 0.014099674432299;
    h_ProbNaturalDeath[22] = 0.014711957238925;
    h_ProbNaturalDeath[23] = 0.014968691990926;
    h_ProbNaturalDeath[24] = 0.014967172185325;
    h_ProbNaturalDeath[25] = 0.014845031894506;
    h_ProbNaturalDeath[26] = 0.014767082029733;
    h_ProbNaturalDeath[27] = 0.01480839152999;
    h_ProbNaturalDeath[28] = 0.015055101638628;
    h_ProbNaturalDeath[29] = 0.015468505190105;
    h_ProbNaturalDeath[30] = 0.015944306819536;
    h_ProbNaturalDeath[31] = 0.016421043709446;
    h_ProbNaturalDeath[32] = 0.016942117099386;
    h_ProbNaturalDeath[33] = 0.017501078390588;
    h_ProbNaturalDeath[34] = 0.018119607730054;
    h_ProbNaturalDeath[35] = 0.018840382831522;
    h_ProbNaturalDeath[36] = 0.01968584321183;
    h_ProbNaturalDeath[37] = 0.020648697917729;
    h_ProbNaturalDeath[38] = 0.02174025422673;
    h_ProbNaturalDeath[39] = 0.022977134791163;
    h_ProbNaturalDeath[40] = 0.024351327962323;
    h_ProbNaturalDeath[41] = 0.025902332950567;
    h_ProbNaturalDeath[42] = 0.02769484883125;
    h_ProbNaturalDeath[43] = 0.029763807831847;
    h_ProbNaturalDeath[44] = 0.032092257416785;
    h_ProbNaturalDeath[45] = 0.03464417578323;
    h_ProbNaturalDeath[46] = 0.037387253883753;
    h_ProbNaturalDeath[47] = 0.040325116908918;
    h_ProbNaturalDeath[48] = 0.043450465802503;
    h_ProbNaturalDeath[49] = 0.046783769250641;
    h_ProbNaturalDeath[50] = 0.050379305647723;
    h_ProbNaturalDeath[51] = 0.054250708597469;
    h_ProbNaturalDeath[52] = 0.058370165793835;
    h_ProbNaturalDeath[53] = 0.062743356857589;
    h_ProbNaturalDeath[54] = 0.067405733722802;
    h_ProbNaturalDeath[55] = 0.072471872691788;
    h_ProbNaturalDeath[56] = 0.077940570749002;
    h_ProbNaturalDeath[57] = 0.083722432777535;
    h_ProbNaturalDeath[58] = 0.089812939592727;
    h_ProbNaturalDeath[59] = 0.096320545551754;
    h_ProbNaturalDeath[60] = 0.103374758030823;
    h_ProbNaturalDeath[61] = 0.111153832260191;
    h_ProbNaturalDeath[62] = 0.119799418763301;
    h_ProbNaturalDeath[63] = 0.129458206950975;
    h_ProbNaturalDeath[64] = 0.140179867078005;
    h_ProbNaturalDeath[65] = 0.151763459015659;
    h_ProbNaturalDeath[66] = 0.164404993288517;
    h_ProbNaturalDeath[67] = 0.178637575194583;
    h_ProbNaturalDeath[68] = 0.194752162647952;
    h_ProbNaturalDeath[69] = 0.212709619046675;
    h_ProbNaturalDeath[70] = 0.232085117364568;
    h_ProbNaturalDeath[71] = 0.25292358002509;
    h_ProbNaturalDeath[72] = 0.27584171899225;
    h_ProbNaturalDeath[73] = 0.301132600796412;
    h_ProbNaturalDeath[74] = 0.328832323976937;
    h_ProbNaturalDeath[75] = 0.358580946581084;
    h_ProbNaturalDeath[76] = 0.390547125436947;
    h_ProbNaturalDeath[77] = 0.425516392727306;
    h_ProbNaturalDeath[78] = 0.463966442471834;
    h_ProbNaturalDeath[79] = 0.506035659978245;

    // Fill values from 80 to 120 with 1.0
    for (int i = 80; i <= 120; i++) {
        h_ProbNaturalDeath[i] = 1.0;
    }

    // Moderate Cases
    for (int i = 0; i < 60; i++)
        h_ProbRecoveryModerate[i] = ProbRecoveryModerateYounger;
    for (int i = 60; i < 70; i++)
        h_ProbRecoveryModerate[i] = ProbRecoveryModerate_60_70;
    for (int i = 70; i < 80; i++)
        h_ProbRecoveryModerate[i] = ProbRecoveryModerate_70_80;
    for (int i = 80; i < 90; i++)
        h_ProbRecoveryModerate[i] = ProbRecoveryModerate_80_90;
    for (int i = 90; i < 121; i++)
        h_ProbRecoveryModerate[i] = ProbRecoveryModerate_Greater90;

    // Severe Cases
    for (int i = 0; i < 60; i++)
        h_ProbRecoverySevere[i] = ProbRecoverySevereYounger;
    for (int i = 60; i < 70; i++)
        h_ProbRecoverySevere[i] = ProbRecoverySevere_60_70;
    for (int i = 70; i < 80; i++)
        h_ProbRecoverySevere[i] = ProbRecoverySevere_70_80;
    for (int i = 80; i < 90; i++)
        h_ProbRecoverySevere[i] = ProbRecoverySevere_80_90;
    for (int i = 90; i < 121; i++)
        h_ProbRecoverySevere[i] = ProbRecoverySevere_Greater90;

    // Hospital Cases
    for (int i = 0; i < 20; i++)
        h_ProbRecoveryH[i] = ProbRecoveryHYounger;
    for (int i = 20; i < 30; i++)
        h_ProbRecoveryH[i] = ProbRecoveryH_20_30;
    for (int i = 30; i < 40; i++)
        h_ProbRecoveryH[i] = ProbRecoveryH_30_40;
    for (int i = 40; i < 50; i++)
        h_ProbRecoveryH[i] = ProbRecoveryH_40_50;
    for (int i = 50; i < 60; i++)
        h_ProbRecoveryH[i] = ProbRecoveryH_50_60;
    for (int i = 60; i < 70; i++)
        h_ProbRecoveryH[i] = ProbRecoveryH_60_70;
    for (int i = 70; i < 80; i++)
        h_ProbRecoveryH[i] = ProbRecoveryH_70_80;
    for (int i = 80; i < 90; i++)
        h_ProbRecoveryH[i] = ProbRecoveryH_80_90;
    for (int i = 90; i < 121; i++)
        h_ProbRecoveryH[i] = ProbRecoveryH_Greater90;


    // ICU Cases
    for (int i = 0; i < 60; i++)
        h_ProbRecoveryICU[i] = ProbRecoveryICUYounger;
    for (int i = 60; i < 70; i++)
        h_ProbRecoveryICU[i] = ProbRecoveryICU_60_70;
    for (int i = 70; i < 80; i++)
        h_ProbRecoveryICU[i] = ProbRecoveryICU_70_80;
    for (int i = 80; i < 90; i++)
        h_ProbRecoveryICU[i] = ProbRecoveryICU_80_90;
    for (int i = 90; i < 121; i++)
        h_ProbRecoveryICU[i] = ProbRecoveryICU_Greater90;

    // Initialize age structure arrays
    
    h_ProbBirthAge[1] = 0.06960000;     // 0–4 years
    h_ProbBirthAge[2] = 0.06920000;     // 5–9 years
    h_ProbBirthAge[3] = 0.06990000;     // 10–14 years
    h_ProbBirthAge[4] = 0.07460000;     // 15–19 years
    h_ProbBirthAge[5] = 0.08140000;     // 20–24 years
    h_ProbBirthAge[6] = 0.08020000;     // 25–29 years
    h_ProbBirthAge[7] = 0.08130000;     // 30–34 years
    h_ProbBirthAge[8] = 0.08040000;     // 35–39 years
    h_ProbBirthAge[9] = 0.07370000;     // 40–44 years
    h_ProbBirthAge[10] = 0.06450000;    // 45–49 years
    h_ProbBirthAge[11] = 0.05960000;    // 50–54 years
    h_ProbBirthAge[12] = 0.05320000;    // 55–59 years
    h_ProbBirthAge[13] = 0.04430000;    // 60–64 years
    h_ProbBirthAge[14] = 0.03470000;    // 65–69 years
    h_ProbBirthAge[15] = 0.02550000;    // 70–74 years
    h_ProbBirthAge[16] = 0.01710000;    // 75–79 years
    h_ProbBirthAge[17] = 0.01120000;    // 80–84 years
    h_ProbBirthAge[18] = 0.00590000;    // 85–89 years
    h_ProbBirthAge[19] = 0.00380000;    // 90+ years

    // Calculate cumulative probabilities
   
    h_SumProbBirthAge[0] = h_ProbBirthAge[1];
    for (int i = 1; i < 19; i++) {
        h_SumProbBirthAge[i] = h_SumProbBirthAge[i - 1] + h_ProbBirthAge[i + 1];
    }

    // Fill age ranges
    h_AgeMin[0] = 0;   h_AgeMax[0] = 4;
    h_AgeMin[1] = 5;   h_AgeMax[1] = 9;
    h_AgeMin[2] = 10;  h_AgeMax[2] = 14;
    h_AgeMin[3] = 15;  h_AgeMax[3] = 19;
    h_AgeMin[4] = 20;  h_AgeMax[4] = 24;
    h_AgeMin[5] = 25;  h_AgeMax[5] = 29;
    h_AgeMin[6] = 30;  h_AgeMax[6] = 34;
    h_AgeMin[7] = 35;  h_AgeMax[7] = 39;
    h_AgeMin[8] = 40;  h_AgeMax[8] = 44;
    h_AgeMin[9] = 45;  h_AgeMax[9] = 49;
    h_AgeMin[10] = 50; h_AgeMax[10] = 54;
    h_AgeMin[11] = 55; h_AgeMax[11] = 59;
    h_AgeMin[12] = 60; h_AgeMax[12] = 64;
    h_AgeMin[13] = 65; h_AgeMax[13] = 69;
    h_AgeMin[14] = 70; h_AgeMax[14] = 74;
    h_AgeMin[15] = 75; h_AgeMax[15] = 79;
    h_AgeMin[16] = 80; h_AgeMax[16] = 84;
    h_AgeMin[17] = 85; h_AgeMax[17] = 89;
    h_AgeMin[18] = 90; h_AgeMax[18] = 100;


}

// Function to initialize GPU constants
__host__ void setupGPUConstants() {
    // Disease States
    cudaMemcpyToSymbol(d_S, &S, sizeof(int));
    cudaMemcpyToSymbol(d_E, &E, sizeof(int));
    cudaMemcpyToSymbol(d_IP, &IP, sizeof(int));
    cudaMemcpyToSymbol(d_IA, &IA, sizeof(int));
    cudaMemcpyToSymbol(d_IS, &IS, sizeof(int));
    cudaMemcpyToSymbol(d_H, &H, sizeof(int));
    cudaMemcpyToSymbol(d_ICU, &ICU, sizeof(int));
    cudaMemcpyToSymbol(d_Recovered, &Recovered, sizeof(int));
    cudaMemcpyToSymbol(d_DeadCovid, &DeadCovid, sizeof(int));
    cudaMemcpyToSymbol(d_Dead, &Dead, sizeof(int));
    cudaMemcpyToSymbol(d_ISLight, &ISLight, sizeof(int));
    cudaMemcpyToSymbol(d_ISModerate, &ISModerate, sizeof(int));
    cudaMemcpyToSymbol(d_ISSevere, &ISSevere, sizeof(int));

    // Simulation Constants
    cudaMemcpyToSymbol(d_L, &L, sizeof(int));
    cudaMemcpyToSymbol(d_N, &N, sizeof(int));
    cudaMemcpyToSymbol(d_DAYS, &DAYS, sizeof(int));
    cudaMemcpyToSymbol(d_MAXSIM, &MAXSIM, sizeof(int));

    // Initial values
    cudaMemcpyToSymbol(d_Eini, &Eini, sizeof(int));
    cudaMemcpyToSymbol(d_IPini, &IPini, sizeof(int));
    cudaMemcpyToSymbol(d_IAini, &IAini, sizeof(int));
    cudaMemcpyToSymbol(d_ISLightini, &ISLightini, sizeof(int));
    cudaMemcpyToSymbol(d_ISModerateini, &ISModerateini, sizeof(int));
    cudaMemcpyToSymbol(d_ISSevereini, &ISSevereini, sizeof(int));
    cudaMemcpyToSymbol(d_Hini, &Hini, sizeof(int));
    cudaMemcpyToSymbol(d_ICUini, &ICUini, sizeof(int));
    cudaMemcpyToSymbol(d_Recoveredini, &Recoveredini, sizeof(int));

    // Disease Parameters
    cudaMemcpyToSymbol(d_Beta, &Beta, sizeof(double));
    cudaMemcpyToSymbol(d_ProbIPtoIA, &ProbIPtoIA, sizeof(double));
    cudaMemcpyToSymbol(d_ProbHtoICU, &ProbHtoICU, sizeof(double));
    cudaMemcpyToSymbol(d_ProbToBecomeISLight, &ProbToBecomeISLight, sizeof(double));
    cudaMemcpyToSymbol(d_ProbToBecomeISModerate, &ProbToBecomeISModerate, sizeof(double));
    cudaMemcpyToSymbol(d_ProbToBecomeISSevere, &ProbToBecomeISSevere, sizeof(double));
    cudaMemcpyToSymbol(d_ProbISLightToISModerate, &ProbISLightToISModerate, sizeof(double));

    // Time Period Constants
    cudaMemcpyToSymbol(d_MinLatency, &MinLatency, sizeof(double));
    cudaMemcpyToSymbol(d_MaxLatency, &MaxLatency, sizeof(double));
    cudaMemcpyToSymbol(d_MinIA, &MinIA, sizeof(double));
    cudaMemcpyToSymbol(d_MaxIA, &MaxIA, sizeof(double));
    cudaMemcpyToSymbol(d_MinIP, &MinIP, sizeof(double));
    cudaMemcpyToSymbol(d_MaxIP, &MaxIP, sizeof(double));
    cudaMemcpyToSymbol(d_MinISLight, &MinISLight, sizeof(double));
    cudaMemcpyToSymbol(d_MaxISLight, &MaxISLight, sizeof(double));
    cudaMemcpyToSymbol(d_MinISModerate, &MinISModerate, sizeof(double));
    cudaMemcpyToSymbol(d_MaxISModerate, &MaxISModerate, sizeof(double));
    cudaMemcpyToSymbol(d_MinISSevere, &MinISSevere, sizeof(double));
    cudaMemcpyToSymbol(d_MaxISSevere, &MaxISSevere, sizeof(double));
    cudaMemcpyToSymbol(d_MinH, &MinH, sizeof(double));
    cudaMemcpyToSymbol(d_MaxH, &MaxH, sizeof(double));
    cudaMemcpyToSymbol(d_MinICU, &MinICU, sizeof(double));
    cudaMemcpyToSymbol(d_MaxICU, &MaxICU, sizeof(double));

    // Health System Constants
    cudaMemcpyToSymbol(d_AverageOcupationRateBeds, &AverageOcupationRateBeds, sizeof(double));
    cudaMemcpyToSymbol(d_AverageOcupationRateBedsICU, &AverageOcupationRateBedsICU, sizeof(double));

    // City Parameters
    cudaMemcpyToSymbol(d_BEDSPOP, &BEDSPOP, sizeof(double));
    cudaMemcpyToSymbol(d_ICUPOP, &ICUPOP, sizeof(double));
    cudaMemcpyToSymbol(d_MaxRandomContacts, &MaxRandomContacts, sizeof(double));
    cudaMemcpyToSymbol(d_MinRandomContacts, &MinRandomContacts, sizeof(double));
    cudaMemcpyToSymbol(d_Density, &Density, sizeof(int));
    cudaMemcpyToSymbol(HIGH, &HIGH, sizeof(int));
    cudaMemcpyToSymbol(LOW, &LOW, sizeof(int));
    cudaMemcpyToSymbol(d_SP, &SP, sizeof(int));
    cudaMemcpyToSymbol(d_ROC, &ROC, sizeof(int));
    cudaMemcpyToSymbol(d_BRA, &BRA, sizeof(int));
    cudaMemcpyToSymbol(d_MAN, &MAN, sizeof(int));
    cudaMemcpyToSymbol(d_C5, &C5, sizeof(int));
    cudaMemcpyToSymbol(d_C6, &C6, sizeof(int));
    cudaMemcpyToSymbol(d_C7, &C7, sizeof(int));
    

    
    // Isolation Constants
    cudaMemcpyToSymbol(d_IsolationYes, &IsolationYes, sizeof(int));
    cudaMemcpyToSymbol(d_IsolationNo, &IsolationNo, sizeof(int));
    cudaMemcpyToSymbol(ON, &ON, sizeof(int));
    cudaMemcpyToSymbol(OFF, &OFF, sizeof(int));

    // Allocate GPU memory for arrays
    cudaMalloc((void**)&d_ProbNaturalDeath, 121 * sizeof(double));
    cudaMalloc((void**)&d_ProbNaturalDeath, 121 * sizeof(double));
    cudaMalloc((void**)&d_ProbRecoveryModerate, 121 * sizeof(double));
    cudaMalloc((void**)&d_ProbRecoverySevere, 121 * sizeof(double));
    cudaMalloc((void**)&d_ProbRecoveryH, 121 * sizeof(double));
    cudaMalloc((void**)&d_ProbRecoveryICU, 121 * sizeof(double));

    cudaMalloc((void**)&d_ProbBirthAge, 21 * sizeof(double));
    cudaMalloc((void**)&d_SumProbBirthAge, 21 * sizeof(double));
    cudaMalloc((void**)&d_AgeMin, 21 * sizeof(int));
    cudaMalloc((void**)&d_AgeMax, 21 * sizeof(int));

    // Initialize probability arrays
    double h_ProbNaturalDeath[121] = { 0 };
    double h_ProbRecoveryModerate[121] = { 0 };
    double h_ProbRecoverySevere[121] = { 0 };
    double h_ProbRecoveryH[121] = { 0 };
    double h_ProbRecoveryICU[121] = { 0 };

    double h_ProbBirthAge[21] = { 0 };
    double h_SumProbBirthAge[21] = { 0 };
    int h_AgeMin[21] = { 0 };
    int h_AgeMax[21] = { 0 };

    // Build arrays
    buildArrays(h_ProbNaturalDeath, h_ProbRecoveryModerate, h_ProbRecoverySevere,
        h_ProbRecoveryH, h_ProbRecoveryICU, h_ProbBirthAge, h_SumProbBirthAge, h_AgeMin, h_AgeMax);

    // Copy to device
    cudaMemcpy(d_ProbNaturalDeath, h_ProbNaturalDeath,
        121 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ProbRecoveryModerate, h_ProbRecoveryModerate,
        121 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ProbRecoverySevere, h_ProbRecoverySevere,
        121 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ProbRecoveryH, h_ProbRecoveryH,
        121 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ProbRecoveryICU, h_ProbRecoveryICU,
        121 * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_ProbBirthAge, h_ProbBirthAge, 21 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_SumProbBirthAge, h_SumProbBirthAge, 21 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_AgeMin, h_AgeMin, 21 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_AgeMax, h_AgeMax, 21 * sizeof(int), cudaMemcpyHostToDevice);
}

// Function to clean up GPU constants
__host__ void cleanupGPUConstants() {
    cudaFree(d_ProbNaturalDeath);
    cudaFree(d_ProbRecoveryModerate);
    cudaFree(d_ProbRecoverySevere);
    cudaFree(d_ProbRecoveryH);
    cudaFree(d_ProbRecoveryICU);
    cudaFree(d_ProbBirthAge);
    cudaFree(d_SumProbBirthAge);
    cudaFree(d_AgeMin);
    cudaFree(d_AgeMax);
}

#endif