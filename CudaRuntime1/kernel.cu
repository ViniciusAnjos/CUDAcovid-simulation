#include<cuda_runtime.h>
#include<stdio.h>
#include<math.h>
#include<stdlib.h>

#include"define.h"

struct Individual
{
	int Health;               // host health state
	int Swap;                 // host health state  to update lattice
	int Gender;
	int AgeYears;
	int AgeDays;
	int AgeDeathYears;
	int AgeDeathDays;
	int StateTime;
	int TimeOnState;
	int Days;
	int Isolation;
	int Exponent; // in case he/he is met by an infectous person
	int Checked;
}Person[L + 2][L + 2];

char nome[30];


/* Data files */
FILE* fp;
FILE* gp;
FILE* hp;
FILE* ip;
FILE* jp;

FILE* rawincidence;
FILE* rawprevalence;



int S_Total;
int E_Total;
int IP_Total;
int IA_Total;
int ISLight_Total;
int ISModerate_Total;
int ISSevere_Total;
int H_Total;
int ICU_Total;
int Recovered_Total;
int DeadCovid_Total;
int Dead_Total;


int New_S;
int New_E;
int New_IP;
int New_IA;
int New_ISLight;
int New_ISModerate;
int New_ISSevere;
int New_H;
int New_ICU;
int New_Recovered;
int New_DeadCovid;
int New_Dead;



/*********************************/

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


double S_Sum[DAYS + 2];
double E_Sum[DAYS + 2];
double IP_Sum[DAYS + 2];
double IA_Sum[DAYS + 2];
double ISLight_Sum[DAYS + 2];
double ISModerate_Sum[DAYS + 2];
double ISSevere_Sum[DAYS + 2];
double H_Sum[DAYS + 2];
double ICU_Sum[DAYS + 2];
double Recovered_Sum[DAYS + 2];
double DeadCovid_Sum[DAYS + 2];
double Dead_Sum[DAYS + 2];



double S_Mean[DAYS + 2];
double E_Mean[DAYS + 2];
double IP_Mean[DAYS + 2];
double IA_Mean[DAYS + 2];
double ISLight_Mean[DAYS + 2];
double ISModerate_Mean[DAYS + 2];
double ISSevere_Mean[DAYS + 2];
double H_Mean[DAYS + 2];
double ICU_Mean[DAYS + 2];
double Recovered_Mean[DAYS + 2];
double DeadCovid_Mean[DAYS + 2];
double Dead_Mean[DAYS + 2];



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



double New_S_Sum[DAYS + 2];
double New_E_Sum[DAYS + 2];
double New_IP_Sum[DAYS + 2];
double New_IA_Sum[DAYS + 2];
double New_ISLight_Sum[DAYS + 2];
double New_ISModerate_Sum[DAYS + 2];
double New_ISSevere_Sum[DAYS + 2];
double New_H_Sum[DAYS + 2];
double New_ICU_Sum[DAYS + 2];
double New_Recovered_Sum[DAYS + 2];
double New_DeadCovid_Sum[DAYS + 2];
double New_Dead_Sum[DAYS + 2];



double New_S_Mean[DAYS + 2];
double New_E_Mean[DAYS + 2];
double New_IP_Mean[DAYS + 2];
double New_IA_Mean[DAYS + 2];
double New_ISLight_Mean[DAYS + 2];
double New_ISModerate_Mean[DAYS + 2];
double New_ISSevere_Mean[DAYS + 2];
double New_H_Mean[DAYS + 2];
double New_ICU_Mean[DAYS + 2];
double New_Recovered_Mean[DAYS + 2];
double New_DeadCovid_Mean[DAYS + 2];
double New_Dead_Mean[DAYS + 2];

double ProbNaturalDeath[121];
double ProbBirthAge[21];
double SumProbBirthAge[21];

double ProbRecoveryModerate[121];
double ProbRecoverySevere[121];
double ProbRecoveryH[121];
double ProbRecoveryICU[121];

int AgeMin[21];
int AgeMax[21];

int time;
int Simulation;
int CountDays;
int Contagion;

unsigned R;
unsigned mult;

double rn;

double CurrentIsolated;
double MAXISOLATED;

char cad[35];

double TotalInfectious;
double TotalInfectiousNew;

int AvailableBeds;
int AvailableBedsICU;

int MaximumIsolated;
int CountIsolated;

double valueR0;

int Density;

double MaxRandomContacts;
double MinRandomContacts;

//double ProportionOfBeds;   // proportion relative to the total population


//double ProportionOfICUBeds;  // proportion relative to the total of hospital beds
//int TotalBedsICU;
double BEDSPOP;
double ICUPOP;

int NumberOfHospitalBeds;
int NumberOfICUBeds;

char nome[30];
char nomeincidence[30];
char nomeprevalence[30];

//////////////////////////////////////////////////////////

double aleat()
{
	R *= mult;
	rn = (double)R / MAXNUM;    /* gerenerate random number [0.0,1.0] */
}
/////////////////////////////////////////////////////////////


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
#include"Update.h"

//CUDA
#include"gpu_define.cuh"

//#include"Image.h"



int main(int argc, char* argv[])
{
	int i, j, t;


	system("rm *.dat");


	gp = fopen("parameters.out", "w");

	fp = fopen("epidemicsprevalence.dat", "w");
	ip = fopen("epidemicsincidence.dat", "w");

	hp = fopen("Infectiousprevalence.dat", "w");
	jp = fopen("Infectiousincidence.dat", "w");

	fprintf(fp, "days\tS_Mean\tE_Mean\tIP_Mean\tIA_Mean\tTotalInfectious\tH_Mean\tICU_Mean\tRecovered_Mean\tDeadCovid_Mean[t]\n");

	fprintf(ip, "New_days\tNew_S_Mean\tNew_E_Mean\tNew_IP_Mean\tNew_IA_Mean\tNew_TotalInfectious\tNew_H_Mean\tNew_ICU_Mean\tNew_Recovered_Mean\tNew_DeadCovid_Mean[t]\n");
	//  R = 893221891;

	mult = 888121;  // for 32 bits compilers

	Agestructute();

	NaturalDeathfunc(); // call the values for natural death probability

	ProbsRecovery();

	cities(ROC);

	for (t = 0; t <= DAYS; t++)
	{
		S_Sum[t] = 0.0;
		E_Sum[t] = 0.0;
		IP_Sum[t] = 0.0;
		IA_Sum[t] = 0.0;
		ISLight_Sum[t] = 0.0;
		ISModerate_Sum[t] = 0.0;
		ISSevere_Sum[t] = 0.0;
		H_Sum[t] = 0.0;
		ICU_Sum[t] = 0.0;
		Recovered_Sum[t] = 0.0;
		DeadCovid_Sum[t] = 0.0;
		Dead_Sum[t] = 0.0;


		New_S_Sum[t] = 0.0;
		New_E_Sum[t] = 0.0;
		New_IP_Sum[t] = 0.0;
		New_IA_Sum[t] = 0.0;
		New_ISLight_Sum[t] = 0.0;
		New_ISModerate_Sum[t] = 0.0;
		New_ISSevere_Sum[t] = 0.0;
		New_H_Sum[t] = 0.0;
		New_ICU_Sum[t] = 0.0;
		New_Recovered_Sum[t] = 0.0;
		New_DeadCovid_Sum[t] = 0.0;
		New_Dead_Sum[t] = 0.0;
	}


	printf("\n beds = %d", NumberOfHospitalBeds);
	printf("\n icu = %d", NumberOfICUBeds);
#if(BeginOfIsolation==NO)
	MaximumIsolated = ProportionIsolated * N;
#else
	MaximumIsolated = 0;
#endif	



	for (Simulation = 1; Simulation <= MAXSIM; Simulation++)
	{
		R = 893221891 * Simulation;

		printf("Simulation=%i\n", Simulation);

		beginfunc();

		CountDays = 0;



		/***** RAW FILES  *****/
		sprintf(nomeincidence, "incidence_%i.dat", Simulation);
		rawincidence = fopen(nomeincidence, "a+");
		fprintf(rawincidence, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", CountDays, New_S_Temp[Simulation][CountDays], New_E_Temp[Simulation][CountDays], New_IP_Temp[Simulation][CountDays],
			New_IA_Temp[Simulation][CountDays], New_ISLight_Temp[Simulation][CountDays], New_ISModerate_Temp[Simulation][CountDays], New_ISSevere_Temp[Simulation][CountDays], New_H_Temp[Simulation][CountDays],
			New_ICU_Temp[Simulation][CountDays], New_Recovered_Temp[Simulation][CountDays], New_DeadCovid_Temp[Simulation][CountDays]);
		fclose(rawincidence);

		sprintf(nomeprevalence, "prevalence_%i.dat", Simulation);
		rawprevalence = fopen(nomeprevalence, "a+");
		fprintf(rawprevalence, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", CountDays, S_TotalTemp[Simulation][CountDays], E_TotalTemp[Simulation][CountDays], IP_TotalTemp[Simulation][CountDays],
			IA_TotalTemp[Simulation][CountDays], ISLight_TotalTemp[Simulation][CountDays], ISModerate_TotalTemp[Simulation][CountDays], ISSevere_TotalTemp[Simulation][CountDays], H_TotalTemp[Simulation][CountDays],
			ICU_TotalTemp[Simulation][CountDays], Recovered_TotalTemp[Simulation][CountDays], DeadCovid_TotalTemp[Simulation][CountDays]);
		fclose(rawprevalence);


		/************************************************/


		AvailableBeds = NumberOfHospitalBeds - NumberOfHospitalBeds * AverageOcupationRateBeds; // Considerind beds occupied due to other diseases

		AvailableBedsICU = NumberOfICUBeds - NumberOfICUBeds * AverageOcupationRateBedsICU;



		for (time = 0; time <= DAYS; time++)
		{
			for (i = 1; i <= L; i++) /* Periodic boudandry conditions */
			{
				Person[0][i].Health = Person[L][i].Health;
				Person[L + 1][i].Health = Person[1][i].Health;

				Person[i][0].Health = Person[i][L].Health;
				Person[i][L + 1].Health = Person[i][1].Health;
			}

			Person[0][0].Health = Person[L][L].Health;  /* Periodic boudandry conditions in the borders */
			Person[0][L + 1].Health = Person[L][1].Health;
			Person[L + 1][0].Health = Person[1][L].Health;
			Person[L + 1][L + 1].Health = Person[1][1].Health;


			for (i = 1; i <= L; i++)
				for (j = 1; j <= L; j++)
					if (Person[i][j].Health == S)
						Sfunc(i, j);
					else if (Person[i][j].Health == E)
						Efunc(i, j);
					else if (Person[i][j].Health == IP)
						IPfunc(i, j);
					else if (Person[i][j].Health == IA || Person[i][j].Health == ISLight || Person[i][j].Health == ISModerate || Person[i][j].Health == ISSevere)
						ISfunc(i, j);
					else if (Person[i][j].Health == H)
						Hfunc(i, j);
					else if (Person[i][j].Health == ICU)
						ICUfunc(i, j);
					else if (Person[i][j].Health == Recovered)
						Person[i][j].Swap = Recovered;

			Updatefunc();   /* Update lattice */

		} // for time


	} // for simulation

	//printf("MaximumIsolated=%i end of simulations\n",MaximumIsolated);

	for (t = 0; t <= DAYS; t++)
	{
		S_Mean[t] = S_Sum[t] / ((double)MAXSIM);
		E_Mean[t] = E_Sum[t] / ((double)MAXSIM);
		IP_Mean[t] = IP_Sum[t] / ((double)MAXSIM);
		IA_Mean[t] = IA_Sum[t] / ((double)MAXSIM);
		ISLight_Mean[t] = ISLight_Sum[t] / ((double)MAXSIM);
		ISModerate_Mean[t] = ISModerate_Sum[t] / ((double)MAXSIM);
		ISSevere_Mean[t] = ISSevere_Sum[t] / ((double)MAXSIM);
		H_Mean[t] = H_Sum[t] / ((double)MAXSIM);
		ICU_Mean[t] = ICU_Sum[t] / ((double)MAXSIM);
		Recovered_Mean[t] = Recovered_Sum[t] / ((double)MAXSIM);
		DeadCovid_Mean[t] = DeadCovid_Sum[t] / ((double)MAXSIM);
		Dead_Mean[t] = Dead_Sum[t] / ((double)MAXSIM);

		New_S_Mean[t] = New_S_Sum[t] / ((double)MAXSIM);
		New_E_Mean[t] = New_E_Sum[t] / ((double)MAXSIM);
		New_IP_Mean[t] = New_IP_Sum[t] / ((double)MAXSIM);
		New_IA_Mean[t] = New_IA_Sum[t] / ((double)MAXSIM);
		New_ISLight_Mean[t] = New_ISLight_Sum[t] / ((double)MAXSIM);
		New_ISModerate_Mean[t] = New_ISModerate_Sum[t] / ((double)MAXSIM);
		New_ISSevere_Mean[t] = New_ISSevere_Sum[t] / ((double)MAXSIM);
		New_H_Mean[t] = New_H_Sum[t] / ((double)MAXSIM);
		New_ICU_Mean[t] = New_ICU_Sum[t] / ((double)MAXSIM);
		New_Recovered_Mean[t] = New_Recovered_Sum[t] / ((double)MAXSIM);
		New_DeadCovid_Mean[t] = New_DeadCovid_Sum[t] / ((double)MAXSIM);
		New_Dead_Mean[t] = New_Dead_Sum[t] / ((double)MAXSIM);

		TotalInfectious = ISLight_Mean[t] + ISModerate_Mean[t] + ISSevere_Mean[t];
		fprintf(hp, "%d\t%f\t%f\t%f\n", t, ISLight_Mean[t], ISModerate_Mean[t], ISSevere_Mean[t]); // prevalevnce of infectious

		TotalInfectiousNew = New_ISLight_Mean[t] + New_ISModerate_Mean[t] + New_ISSevere_Mean[t];
		fprintf(jp, "%d\t%f\t%f\t%f\n", t, New_ISLight_Mean[t], New_ISModerate_Mean[t], New_ISSevere_Mean[t]); // incidence of infectious

		/* incidence */
		fprintf(ip, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", t, New_S_Mean[t], New_E_Mean[t], New_IP_Mean[t], New_IA_Mean[t], TotalInfectiousNew, New_H_Mean[t], New_ICU_Mean[t], New_Recovered_Mean[t], New_DeadCovid_Mean[t]);

		//printf("%i\t%f\n",t,N*DeadCovid_Mean[t]);

		/* prevalence */
		fprintf(fp, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", t, S_Mean[t], E_Mean[t], IP_Mean[t], IA_Mean[t], TotalInfectious, H_Mean[t], ICU_Mean[t], Recovered_Mean[t], DeadCovid_Mean[t]);



	} // t=0,t<DAY
	system("cp epidemicsprevalence.dat /mnt/c/Users/vinih/Projetos_python/Modelagem_covid/dados");
	system("cp epidemicsincidence.dat /mnt/c/Users/vinih/Projetos_python/Modelagem_covid/dados");
	return 0;
}
