/* Covid states */
const int S = 1;             /* Susceptible individuals */
const int E = 2;             /* Pre symptomatic */
const int IP = 3;            /* LS individual already submitted to exogenous reinfection */
const int IA = 4;            /* Infectious individuals asymptomatic */
const int IS = 5;            /* Infectious individuals Symptomatic  */
const int H = 6;             /* HospiTalized inidividual (ISSevere) */
const int ICU = 7;           /* Individiual in ICU (ISSevere) */
const int Recovered = 8;     /* Recovered from COVID-19 */
const int DeadCovid = 9;     /* Dead due to covid */
const int Dead = 10;         /* Dead due to natutal death */

/* Sub states */
const int ISLight = 12;
const int ISModerate = 13;
const int ISSevere = 14;

/* For random number generator */
const double MAXNUM = 4294967295.;             /* for 32 bits compilers */

const int DAYS = 400;        /* Number of days simulated */
const int MAXSIM = 1;        /* Number of simulations to evaluate averages */

const int L = 3200;
const int N = L * L;

//cities
const int SP = 1;
const int ROC = 2;
const int BRA = 3;
const int MAN = 4;
const int C5 = 5;
const int C6 = 6;
const int C7 = 7;

const double ProportionIsolated = 0.5;
const int BeginOfIsolation = 0;  // OFF - time to begin isolation
const int TimeTiggerIsolation = 15; // days

const int IsolationYes = 1;
const int IsolationNo = 0;

/** Beta for 1000 average and r0 = 3.5**/
const double Beta = 0.3;

const double AverageOcupationRateBeds = 0.5;     // average ocupation rate of hospital beds due to others diseases
const double AverageOcupationRateBedsICU = 0.5;  // average ocupation rate of hospital beds due to others diseases

/* Population parameters  - Initializion values */
const int Eini = 0;
const int IPini = 5;
const int IAini = 0;
const int ISLightini = 0;
const int ISModerateini = 0;
const int ISSevereini = 0;
const int Hini = 0;
const int ICUini = 0;
const int Recoveredini = 0;

/* Disease parameters */
/* Probs of evolution between states */
const double ProbIPtoIA = 0.5;
const double ProbHtoICU = 0.25;
const double ProbToBecomeISLight = 0.6;
const double ProbToBecomeISModerate = 0.2;
const double ProbToBecomeISSevere = 0.2;
const double ProbISLightToISModerate = 0.10;  /// NEW CHECK CHECK

/* Probs of Recovery < 60 */
const double ProbRecoveryModerateYounger = 0.6;
const double ProbRecoverySevereYounger = 0.01;   // out of hospital
const double ProbRecoveryHYounger = 1.0;   ///NEW NEW < 20
const double ProbRecoveryICUYounger = 0.5;

/* Probs Recovery IS Moderate for Elerdely */
const double ProbRecoveryModerate_60_70 = 0.21;
const double ProbRecoveryModerate_70_80 = 0.15;
const double ProbRecoveryModerate_80_90 = 0.13;
const double ProbRecoveryModerate_Greater90 = 0.10;

/* Probs Recovery IS Severe for Elerdely */
const double ProbRecoverySevere_60_70 = 0.00357;
const double ProbRecoverySevere_70_80 = 0.00250;
const double ProbRecoverySevere_80_90 = 0.00125;
const double ProbRecoverySevere_Greater90 = 0.00167;

/* Probs Recovery H for age > 20 */
const double ProbRecoveryH_20_30 = 0.959;
const double ProbRecoveryH_30_40 = 0.962;
const double ProbRecoveryH_40_50 = 0.938;
const double ProbRecoveryH_50_60 = 0.897;
const double ProbRecoveryH_60_70 = 0.842;
const double ProbRecoveryH_70_80 = 0.678;
const double ProbRecoveryH_80_90 = 0.457;
const double ProbRecoveryH_Greater90 = 0.477;

/* Probs Recovery ICU for Elerdely */
const double ProbRecoveryICU_60_70 = 0.17857;
const double ProbRecoveryICU_70_80 = 0.12500;
const double ProbRecoveryICU_80_90 = 0.10417;
const double ProbRecoveryICU_Greater90 = 0.08333;

/*  Periods on states */
const double MinLatency = 0.0;           // days
const double MaxLatency = 27.5;         // days

const double MinIA = 0.0;          // days
const double MaxIA = 7.5;        // days

const double MinIP = 0.0;          // days
const double MaxIP = 14.5;        // days

const double MinISLight = 0.0;          // days
const double MaxISLight = 14.5;        // days

const double MinISModerate = 0.0;          // days
const double MaxISModerate = 28.0;        // days

const double MinISSevere = 0.0;          // days
const double MaxISSevere = 4.0;        // days

const double MinH = 7.5;
const double MaxH = 45.5;

const double MinICU = 10.5;
const double MaxICU = 60.5;

const int ON = 1;
const int OFF = 0;

const int LOW = 0;
const int HIGH = 1;


double BEDSPOP;
double ICUPOP;

int NumberOfHospitalBeds;
int NumberOfICUBeds;


int MaximumIsolated;
int CountIsolated;

double valueR0;

int Density;

double MaxRandomContacts;
double MinRandomContacts;

//double ProportionOfBeds;   // proportion relative to the total population


//double ProportionOfICUBeds;  // proportion relative to the total of hospital beds
//int TotalBedsICU;

