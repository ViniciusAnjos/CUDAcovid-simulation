// output_files.cuh - Complete file output system matching original C code


// File pointers for output files (matching original covid.c)
FILE* fp;                    // epidemicsprevalence.dat
FILE* ip;                    // epidemicsincidence.dat  
FILE* hp;                    // Infectiousprevalence.dat
FILE* jp;                    // Infectiousincidence.dat
FILE* rawincidence;          // incidence_[sim].dat
FILE* rawprevalence;         // prevalence_[sim].dat

// File names for per-simulation raw data
char nomeincidence[30];
char nomeprevalence[30];

// Variables for infectious totals (matching original)
double TotalInfectious;
double TotalInfectiousNew;

// Host function to initialize output files
void initializeOutputFiles() {
    // Remove any existing data files (matching original)
    #ifdef _WIN32
        system("del *.dat");
    #else
        system("rm *.dat");
    #endif

    // Open main output files (matching original covid.c)
    fp = fopen("epidemicsprevalence.dat", "w");
    ip = fopen("epidemicsincidence.dat", "w");
    hp = fopen("Infectiousprevalence.dat", "w");
    jp = fopen("Infectiousincidence.dat", "w");

    // Write headers to main files (matching original format)
    fprintf(fp, "days\tS_Mean\tE_Mean\tIP_Mean\tIA_Mean\tTotalInfectious\tH_Mean\tICU_Mean\tRecovered_Mean\tDeadCovid_Mean\n");
    fprintf(ip, "New_days\tNew_S_Mean\tNew_E_Mean\tNew_IP_Mean\tNew_IA_Mean\tNew_TotalInfectious\tNew_H_Mean\tNew_ICU_Mean\tNew_Recovered_Mean\tNew_DeadCovid_Mean\n");
}

// Host function to write initial simulation data (Day 0)
void writeInitialSimulationData(int simulation, int* h_totals, int* h_new_cases, int N) {
    // Calculate proportions (matching original begin.h logic)
    double S_TotalTemp = (double)h_totals[S] / (double)N;
    double E_TotalTemp = (double)h_totals[E] / (double)N;
    double IP_TotalTemp = (double)h_totals[IP] / (double)N;
    double IA_TotalTemp = (double)h_totals[IA] / (double)N;
    double ISLight_TotalTemp = (double)h_totals[ISLight] / (double)N;
    double ISModerate_TotalTemp = (double)h_totals[ISModerate] / (double)N;
    double ISSevere_TotalTemp = (double)h_totals[ISSevere] / (double)N;
    double H_TotalTemp = (double)h_totals[H] / (double)N;
    double ICU_TotalTemp = (double)h_totals[ICU] / (double)N;
    double Recovered_TotalTemp = (double)h_totals[Recovered] / (double)N;
    double DeadCovid_TotalTemp = (double)h_totals[DeadCovid] / (double)N;

    double New_S_Temp = (double)h_new_cases[S] / (double)N;
    double New_E_Temp = (double)h_new_cases[E] / (double)N;
    double New_IP_Temp = (double)h_new_cases[IP] / (double)N;
    double New_IA_Temp = (double)h_new_cases[IA] / (double)N;
    double New_ISLight_Temp = (double)h_new_cases[ISLight] / (double)N;
    double New_ISModerate_Temp = (double)h_new_cases[ISModerate] / (double)N;
    double New_ISSevere_Temp = (double)h_new_cases[ISSevere] / (double)N;
    double New_H_Temp = (double)h_new_cases[H] / (double)N;
    double New_ICU_Temp = (double)h_new_cases[ICU] / (double)N;
    double New_Recovered_Temp = (double)h_new_cases[Recovered] / (double)N;
    double New_DeadCovid_Temp = (double)h_new_cases[DeadCovid] / (double)N;

    // Open raw data files for this simulation (matching original)
    sprintf(nomeincidence, "incidence_%d.dat", simulation);
    rawincidence = fopen(nomeincidence, "a+");
    fprintf(rawincidence, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
        0, New_S_Temp, New_E_Temp, New_IP_Temp, New_IA_Temp,
        New_ISLight_Temp, New_ISModerate_Temp, New_ISSevere_Temp,
        New_H_Temp, New_ICU_Temp, New_Recovered_Temp, New_DeadCovid_Temp);
    fclose(rawincidence);

    sprintf(nomeprevalence, "prevalence_%d.dat", simulation);
    rawprevalence = fopen(nomeprevalence, "a+");
    fprintf(rawprevalence, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
        0, S_TotalTemp, E_TotalTemp, IP_TotalTemp, IA_TotalTemp,
        ISLight_TotalTemp, ISModerate_TotalTemp, ISSevere_TotalTemp,
        H_TotalTemp, ICU_TotalTemp, Recovered_TotalTemp, DeadCovid_TotalTemp);
    fclose(rawprevalence);
}

// Host function to write daily simulation data
void writeDailySimulationData(int simulation, int day, int* h_totals, int* h_new_cases, int N) {
    // Calculate proportions (matching original Update.h logic)
    double S_TotalTemp = (double)h_totals[S] / (double)N;
    double E_TotalTemp = (double)h_totals[E] / (double)N;
    double IP_TotalTemp = (double)h_totals[IP] / (double)N;
    double IA_TotalTemp = (double)h_totals[IA] / (double)N;
    double ISLight_TotalTemp = (double)h_totals[ISLight] / (double)N;
    double ISModerate_TotalTemp = (double)h_totals[ISModerate] / (double)N;
    double ISSevere_TotalTemp = (double)h_totals[ISSevere] / (double)N;
    double H_TotalTemp = (double)h_totals[H] / (double)N;
    double ICU_TotalTemp = (double)h_totals[ICU] / (double)N;
    double Recovered_TotalTemp = (double)h_totals[Recovered] / (double)N;
    double DeadCovid_TotalTemp = (double)h_totals[DeadCovid] / (double)N;

    double New_S_Temp = (double)h_new_cases[S] / (double)N;
    double New_E_Temp = (double)h_new_cases[E] / (double)N;
    double New_IP_Temp = (double)h_new_cases[IP] / (double)N;
    double New_IA_Temp = (double)h_new_cases[IA] / (double)N;
    double New_ISLight_Temp = (double)h_new_cases[ISLight] / (double)N;
    double New_ISModerate_Temp = (double)h_new_cases[ISModerate] / (double)N;
    double New_ISSevere_Temp = (double)h_new_cases[ISSevere] / (double)N;
    double New_H_Temp = (double)h_new_cases[H] / (double)N;
    double New_ICU_Temp = (double)h_new_cases[ICU] / (double)N;
    double New_Recovered_Temp = (double)h_new_cases[Recovered] / (double)N;
    double New_DeadCovid_Temp = (double)h_new_cases[DeadCovid] / (double)N;

    // Write to raw data files for this simulation (matching original Update.h)
    sprintf(nomeincidence, "incidence_%d.dat", simulation);
    rawincidence = fopen(nomeincidence, "a+");
    fprintf(rawincidence, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
        day, New_S_Temp, New_E_Temp, New_IP_Temp, New_IA_Temp,
        New_ISLight_Temp, New_ISModerate_Temp, New_ISSevere_Temp,
        New_H_Temp, New_ICU_Temp, New_Recovered_Temp, New_DeadCovid_Temp);
    fclose(rawincidence);

    sprintf(nomeprevalence, "prevalence_%d.dat", simulation);
    rawprevalence = fopen(nomeprevalence, "a+");
    fprintf(rawprevalence, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
        day, S_TotalTemp, E_TotalTemp, IP_TotalTemp, IA_TotalTemp,
        ISLight_TotalTemp, ISModerate_TotalTemp, ISSevere_TotalTemp,
        H_TotalTemp, ICU_TotalTemp, Recovered_TotalTemp, DeadCovid_TotalTemp);
    fclose(rawprevalence);
}

// Host function to write final averaged results (matching original covid.c end)
void writeFinalAveragedResults(double* S_Mean, double* E_Mean, double* IP_Mean,
    double* IA_Mean, double* ISLight_Mean, double* ISModerate_Mean,
    double* ISSevere_Mean, double* H_Mean, double* ICU_Mean,
    double* Recovered_Mean, double* DeadCovid_Mean,
    double* New_S_Mean, double* New_E_Mean, double* New_IP_Mean,
    double* New_IA_Mean, double* New_ISLight_Mean, double* New_ISModerate_Mean,
    double* New_ISSevere_Mean, double* New_H_Mean, double* New_ICU_Mean,
    double* New_Recovered_Mean, double* New_DeadCovid_Mean,
    int DAYS) {

    // Write final results (matching original covid.c logic)
    for (int t = 0; t <= DAYS; t++) {
        // Calculate total infectious (matching original)
        TotalInfectious = ISLight_Mean[t] + ISModerate_Mean[t] + ISSevere_Mean[t];
        TotalInfectiousNew = New_ISLight_Mean[t] + New_ISModerate_Mean[t] + New_ISSevere_Mean[t];

        // Write to infectious files (matching original)
        fprintf(hp, "%d\t%f\t%f\t%f\n", t, ISLight_Mean[t], ISModerate_Mean[t], ISSevere_Mean[t]);
        fprintf(jp, "%d\t%f\t%f\t%f\n", t, New_ISLight_Mean[t], New_ISModerate_Mean[t], New_ISSevere_Mean[t]);

        // Write incidence data (matching original format)
        fprintf(ip, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", t,
            New_S_Mean[t], New_E_Mean[t], New_IP_Mean[t], New_IA_Mean[t],
            TotalInfectiousNew, New_H_Mean[t], New_ICU_Mean[t],
            New_Recovered_Mean[t], New_DeadCovid_Mean[t]);

        // Write prevalence data (matching original format)
        fprintf(fp, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", t,
            S_Mean[t], E_Mean[t], IP_Mean[t], IA_Mean[t],
            TotalInfectious, H_Mean[t], ICU_Mean[t],
            Recovered_Mean[t], DeadCovid_Mean[t]);
    }
}

// Host function to close all output files
void closeOutputFiles() {
    fclose(fp);
    fclose(ip);
    fclose(hp);
    fclose(jp);

    printf("\nOutput files generated:\n");
    printf("- epidemicsprevalence.dat\n");
    printf("- epidemicsincidence.dat\n");
    printf("- Infectiousprevalence.dat\n");
    printf("- Infectiousincidence.dat\n");
    printf("- prevalence_[1-MAXSIM].dat\n");
    printf("- incidence_[1-MAXSIM].dat\n");
}