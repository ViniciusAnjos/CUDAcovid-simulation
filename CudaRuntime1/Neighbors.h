int Neighborsfunc(int i, int j)
{

	int k, l, m, n;
	int KI; /* counter for infected individuals in the neighborhood */
	int RandomContacts;
	int Randomi;
	int Randomj;
	int mute;

	double ProbContagion;

	Contagion = 0;

	KI = 0;

	if (Person[i][j].Isolation == IsolationNo) // isolation only for random contacts
	{
		aleat();
		RandomContacts = rn * (MaxRandomContacts - MinRandomContacts) + MinRandomContacts;


		mute = 0; /* Check random contacts */

		if (RandomContacts > 0)
			do
			{
				do {
					aleat();
					Randomi = rn * L + 1;
				} while (Randomi == i);

				do {
					aleat();
					Randomj = rn * L + 1;
				} while (Randomj == j);

				if (Person[Randomi][Randomj].Health == IP || Person[Randomi][Randomj].Health == ISLight || Person[Randomi][Randomj].Health == ISModerate ||
					Person[Randomi][Randomj].Health == ISSevere || Person[Randomi][Randomj].Health == IA || Person[Randomi][Randomj].Health == H || Person[Randomi][Randomj].Health == ICU)
					KI++;

				mute++;
			} while (mute < RandomContacts);
	}


#if(Density==HIGH)  // high demographic density
	{
		/* Check 8 neighbors in the lattice */
		for (k = -1; k <= 1; k++)
			for (l = -1; l <= 1; l++)
				if (Person[i + k][j + l].Health == IP || Person[i + k][j + l].Health == ISLight || Person[i + k][j + l].Health == ISModerate || Person[i + k][j + l].Health == ISSevere ||
					Person[i + k][j + l].Health == IA || Person[i + k][j + l].Health == H || Person[i + k][j + l].Health == ICU)
					KI++;
	}
#else
	{


		/* Check 4 neighbors in the lattice */
		if (Person[i - 1][j].Health == IP || Person[i - 1][j].Health == ISLight || Person[i - 1][j].Health == ISModerate || Person[i - 1][j].Health == ISSevere || Person[i - 1][j].Health == IA || Person[i - 1][j].Health == H || Person[i - 1][j].Health == ICU)
			KI++;
		else if (Person[i + 1][j].Health == IP || Person[i + 1][j].Health == ISLight || Person[i + 1][j].Health == ISModerate || Person[i + 1][j].Health == ISSevere || Person[i + 1][j].Health == IA || Person[i + 1][j].Health == H || Person[i + 1][j].Health == ICU)
			KI++;
		else if (Person[i][j - 1].Health == IP || Person[i][j - 1].Health == ISLight || Person[i][j - 1].Health == ISModerate || Person[i][j - 1].Health == ISSevere || Person[i][j - 1].Health == IA || Person[i][j - 1].Health == H || Person[i][j - 1].Health == ICU)
			KI++;
		else if (Person[i][j + 1].Health == IP || Person[i][j + 1].Health == ISLight || Person[i][j + 1].Health == ISModerate || Person[i][j + 1].Health == ISSevere || Person[i][j + 1].Health == IA || Person[i][j + 1].Health == H || Person[i][j + 1].Health == ICU)
			KI++;
	}
#endif

	if (KI > 0) // Calculate ProbContagion
		ProbContagion = 1.0 - pow(1.0 - Beta, (double)KI);
	else
		ProbContagion = 0.0;

	if (KI == 0)
		Contagion = 0; // No contagion
	else
	{
		aleat();
		if (rn <= ProbContagion)
			Contagion = 1;
		else
			Contagion = 0;
	}

	return Contagion;
}