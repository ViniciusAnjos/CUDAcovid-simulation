// Medidor de R0 para a simulacao SERIAL (Sao Paulo, parametros do documento).
// R0 = numero medio de secundarios que 1 paciente-zero infecta diretamente em
// populacao 100% suscetivel, contado enquanto o paciente-zero esta em estado que
// espalha (IP/IA/ISLight) -- mesma definicao da calibracao da GPU (early-stop).
//
// Compilar:
//   nvcc r0_serial.cu -o r0_serial.exe -arch=sm_89 --diag-suppress 20091 ^
//        --diag-suppress 177 --diag-suppress 940 -Xcompiler "/wd4716"
// Requer no define.h: IPini = 1 (1 paciente-zero). L/MAXSIM controlam custo/precisao.
// R0 e independente de L (random contacts ~ contacts/dia, nao de L).

#include<cuda_runtime.h>
#include<stdio.h>
#include<math.h>
#include<stdlib.h>

#include"define.h"

struct Individual
{
    int Health;
    int Swap;
    int Gender;
    int AgeYears;
    int AgeDays;
    int AgeDeathYears;
    int AgeDeathDays;
    int StateTime;
    int TimeOnState;
    int Days;
    int Isolation;
    int Exponent;
    int Checked;
}Person[L + 2][L + 2];

char nome[30];

FILE* fp; FILE* gp; FILE* hp; FILE* ip; FILE* jp;
FILE* rawincidence; FILE* rawprevalence;

int S_Total, E_Total, IP_Total, IA_Total, ISLight_Total, ISModerate_Total, ISSevere_Total;
int H_Total, ICU_Total, Recovered_Total, DeadCovid_Total, Dead_Total;

int New_S, New_E, New_IP, New_IA, New_ISLight, New_ISModerate, New_ISSevere;
int New_H, New_ICU, New_Recovered, New_DeadCovid, New_Dead;

double S_TotalTemp[MAXSIM + 2][DAYS + 2];
double E_TotalTemp[MAXSIM + 2][DAYS + 2];
double IP_TotalTemp[MAXSIM + 2][DAYS + 2];
double IA_TotalTemp[MAXSIM + 2][DAYS + 2];
double ISLight_TotalTemp[MAXSIM + 2][DAYS + 2];
double ISModerate_TotalTemp[MAXSIM + 2][DAYS + 2];
double ISSevere_TotalTemp[MAXSIM + 2][DAYS + 2];
double H_TotalTemp[MAXSIM + 2][DAYS + 2];
double ICU_TotalTemp[MAXSIM + 2][DAYS + 2];
double Recovered_TotalTemp[MAXSIM + 2][DAYS + 2];
double DeadCovid_TotalTemp[MAXSIM + 2][DAYS + 2];
double Dead_TotalTemp[MAXSIM + 2][DAYS + 2];

double New_S_Temp[MAXSIM + 2][DAYS + 2];
double New_E_Temp[MAXSIM + 2][DAYS + 2];
double New_IP_Temp[MAXSIM + 2][DAYS + 2];
double New_IA_Temp[MAXSIM + 2][DAYS + 2];
double New_ISLight_Temp[MAXSIM + 2][DAYS + 2];
double New_ISModerate_Temp[MAXSIM + 2][DAYS + 2];
double New_ISSevere_Temp[MAXSIM + 2][DAYS + 2];
double New_H_Temp[MAXSIM + 2][DAYS + 2];
double New_ICU_Temp[MAXSIM + 2][DAYS + 2];
double New_Recovered_Temp[MAXSIM + 2][DAYS + 2];
double New_DeadCovid_Temp[MAXSIM + 2][DAYS + 2];
double New_Dead_Temp[MAXSIM + 2][DAYS + 2];

// Sum arrays (begin.h referencia [0])
double S_Sum[DAYS + 2], E_Sum[DAYS + 2], IP_Sum[DAYS + 2], IA_Sum[DAYS + 2];
double ISLight_Sum[DAYS + 2], ISModerate_Sum[DAYS + 2], ISSevere_Sum[DAYS + 2];
double H_Sum[DAYS + 2], ICU_Sum[DAYS + 2], Recovered_Sum[DAYS + 2];
double DeadCovid_Sum[DAYS + 2], Dead_Sum[DAYS + 2];
double New_S_Sum[DAYS + 2], New_E_Sum[DAYS + 2], New_IP_Sum[DAYS + 2], New_IA_Sum[DAYS + 2];
double New_ISLight_Sum[DAYS + 2], New_ISModerate_Sum[DAYS + 2], New_ISSevere_Sum[DAYS + 2];
double New_H_Sum[DAYS + 2], New_ICU_Sum[DAYS + 2], New_Recovered_Sum[DAYS + 2];
double New_DeadCovid_Sum[DAYS + 2], New_Dead_Sum[DAYS + 2];

double ProbNaturalDeath[121];
double ProbBirthAge[21];
double SumProbBirthAge[21];
double ProbRecoveryModerate[121];
double ProbRecoverySevere[121];
double ProbRecoveryH[121];
double ProbRecoveryICU[121];
int AgeMin[21];
int AgeMax[21];

int sim_time, Simulation, CountDays, Contagion;
unsigned R, mult;
double rn;
double CurrentIsolated, MAXISOLATED;
char cad[35];
double TotalInfectious, TotalInfectiousNew;
int AvailableBeds, AvailableBedsICU;
char nomeincidence[30], nomeprevalence[30];

double aleat()
{
    R *= mult;
    rn = (double)R / MAXNUM;
}

#include"begin.h"
#include"cities.h"
#include"death.h"
#include"agestructure.h"
#include"probsrecovery.h"
#include"Neighbors.h"
#include"Neighborsinfected.h"
#include"S.h"
#include"E.h"
#include"IP.h"
#include"IS.h"
#include"H.h"
#include"ICU.h"
#include"isolation.h"
// NAO incluir Update.h -- usamos leanUpdate (sem I/O)

// Atualizacao enxuta: Health=Swap, reset Exponent/Checked, envelhece, substitui mortos,
// recconta os Totais. Sem arquivos/Temp/Sum (igual a parte essencial do Updatefunc).
void leanUpdate()
{
    int i, j, mute;
    S_Total = E_Total = IP_Total = IA_Total = ISLight_Total = 0;
    ISModerate_Total = ISSevere_Total = H_Total = ICU_Total = Recovered_Total = Dead_Total = 0;

    for (i = 1; i <= L; i++)
        for (j = 1; j <= L; j++)
        {
            Person[i][j].Health = Person[i][j].Swap;
            Person[i][j].Exponent = 0;
            Person[i][j].Checked = 0;
            Person[i][j].AgeDays++;
            Person[i][j].Days++;

            if (Person[i][j].AgeYears >= Person[i][j].AgeDeathYears)
                Person[i][j].Health = Dead;

            if (Person[i][j].Health == Dead || Person[i][j].Health == DeadCovid)
            {
                Person[i][j].Health = S;
                aleat(); Person[i][j].AgeYears = rn * 100;
                Person[i][j].AgeDays = Person[i][j].AgeYears * 365;
                Person[i][j].Days = 0;
                Person[i][j].TimeOnState = 0;
                Person[i][j].StateTime = 0;
                mute = 0;
                do {
                    aleat(); Person[i][j].AgeDeathYears = rn * 100;
                    aleat();
                    if (rn < ProbNaturalDeath[Person[i][j].AgeDeathYears]) mute = 1; else mute = 0;
                } while (mute < 1);
                Person[i][j].AgeDeathDays = Person[i][j].AgeDeathYears * 365;
                if (Person[i][j].AgeDeathYears < Person[i][j].AgeYears)
                {
                    mute = Person[i][j].AgeDeathYears;
                    Person[i][j].AgeYears = Person[i][j].AgeDeathYears;
                    Person[i][j].AgeDeathYears = mute;
                    Person[i][j].AgeDeathDays = Person[i][j].AgeDeathYears * 365;
                }
            }
        }

    for (i = 1; i <= L; i++)
        for (j = 1; j <= L; j++)
        {
            if (Person[i][j].Health == S) S_Total++;
            else if (Person[i][j].Health == E) E_Total++;
            else if (Person[i][j].Health == IP) IP_Total++;
            else if (Person[i][j].Health == IA) IA_Total++;
            else if (Person[i][j].Health == ISLight) ISLight_Total++;
            else if (Person[i][j].Health == ISModerate) ISModerate_Total++;
            else if (Person[i][j].Health == ISSevere) ISSevere_Total++;
            else if (Person[i][j].Health == H) H_Total++;
            else if (Person[i][j].Health == ICU) ICU_Total++;
            else if (Person[i][j].Health == Recovered) Recovered_Total++;
        }
}

int main(int argc, char* argv[])
{
    int i, j;

    mult = 888121;
    Agestructute();
    NaturalDeathfunc();
    ProbsRecovery();
    cities(SP);

    AvailableBeds = NumberOfHospitalBeds - NumberOfHospitalBeds * AverageOcupationRateBeds;
    AvailableBedsICU = NumberOfICUBeds - NumberOfICUBeds * AverageOcupationRateBedsICU;

    double R0_Sum = 0.0;
    long zeros = 0;

    for (Simulation = 1; Simulation <= MAXSIM; Simulation++)
    {
        R = 893221891 * Simulation;
        beginfunc();   // semeia IPini(=1) IP, resto S; ajusta Totais

        long r0count = 0;

        for (sim_time = 0; sim_time < DAYS; sim_time++)
        {
            // para quando o paciente-zero deixa os estados que espalham (IP/IA/ISLight),
            // ou quando um secundario passa a espalhar (== 2). Antes disso, ele e o
            // UNICO infeccioso, logo todo New_E e filho direto dele.
            int spreading = IP_Total + IA_Total + ISLight_Total;
            if (spreading != 1) break;

            for (i = 1; i <= L; i++) {
                Person[0][i].Health = Person[L][i].Health;
                Person[L + 1][i].Health = Person[1][i].Health;
                Person[i][0].Health = Person[i][L].Health;
                Person[i][L + 1].Health = Person[i][1].Health;
            }
            Person[0][0].Health = Person[L][L].Health;
            Person[0][L + 1].Health = Person[L][1].Health;
            Person[L + 1][0].Health = Person[1][L].Health;
            Person[L + 1][L + 1].Health = Person[1][1].Health;

            New_E = 0;
            for (i = 1; i <= L; i++)
                for (j = 1; j <= L; j++)
                    if (Person[i][j].Health == S) Sfunc(i, j);
                    else if (Person[i][j].Health == E) Efunc(i, j);
                    else if (Person[i][j].Health == IP) IPfunc(i, j);
                    else if (Person[i][j].Health == IA || Person[i][j].Health == ISLight || Person[i][j].Health == ISModerate || Person[i][j].Health == ISSevere) ISfunc(i, j);
                    else if (Person[i][j].Health == H) Hfunc(i, j);
                    else if (Person[i][j].Health == ICU) ICUfunc(i, j);
                    else if (Person[i][j].Health == Recovered) Person[i][j].Swap = Recovered;

            r0count += New_E;
            leanUpdate();
        }

        R0_Sum += (double)r0count;
        if (r0count == 0) zeros++;
    }

    printf("=== R0 SERIAL Sao Paulo (parametros do documento) ===\n");
    printf("L=%d  MAXSIM=%d  contatos[%.1f,%.1f]  Beta=%.5f\n",
        L, MAXSIM, MinRandomContacts, MaxRandomContacts, Beta);
    printf("R0 medio = %.4f   (alvo 3.5;  %% rodadas com 0 secundarios = %.1f%%)\n",
        R0_Sum / (double)MAXSIM, 100.0 * zeros / (double)MAXSIM);
    return 0;
}
