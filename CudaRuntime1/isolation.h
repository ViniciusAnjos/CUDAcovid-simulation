void Isolationfunc()
{
	int counter;
	int counterrepetition; // counter to whteher the lattice was teste more than twice
	int EffectivelyIsolated;
	int IsolatedLeft;
	int Randomi;
	int Randomj;
	int i, j;

	//printf("Acessou Isolation function\n");


	counterrepetition = 0;

	EffectivelyIsolated = 0;
	counter = 0;
	do {
		aleat();
		Randomi = rn * L + 1;

		aleat();
		Randomj = rn * L + 1;


		if (Person[Randomi][Randomj].Isolation == IsolationNo && (Person[Randomi][Randomj].Health != H || Person[Randomi][Randomj].Health != ICU))
		{
			Person[Randomi][Randomj].Isolation = IsolationYes;
			EffectivelyIsolated++;
			counter++;
		}

		counterrepetition++;

		if (counterrepetition == 2 * N)  // if the lattice was checked twide, avoid an infinite loop
			counter = MaximumIsolated;

	} while (counter < MaximumIsolated);

	//printf("EffectivelyIsolated=%i\n",EffectivelyIsolated);

	IsolatedLeft = MaximumIsolated - EffectivelyIsolated;

	//printf("IsolatedLeft=%i\n",IsolatedLeft);


	if (IsolatedLeft > 0)
	{
		counter = 0;
		for (i = 1; i <= L; i++)
			for (j = 1; j <= L; j++)
			{
				aleat();
				if (rn < ProportionIsolated)
				{
					if (Person[i][j].Isolation == IsolationNo && (Person[i][j].Health != H || Person[i][j].Health != ICU))
					{
						Person[i][j].Isolation = IsolationYes;
						counter++;
					}
				}

				if (counter == IsolatedLeft)
					break;
			}
	}

	//if(IsolatedLeft > 0)
	//	printf("Maximum isolated is %i\n\n",EffectivelyIsolated + counter);
	//else
		//printf("Maximum isolated is %i\n\n",EffectivelyIsolated);


}
