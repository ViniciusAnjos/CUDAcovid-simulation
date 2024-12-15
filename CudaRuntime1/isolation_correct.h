void Isolationfunc()
{
	int mute;
	int Randomi;
	int Randomj;


	printf("Acessou Isolation function\n");

	mute = 0;
	do {
		aleat();
		Randomi = rn * L + 1;

		aleat();
		Randomj = rn * L + 1;


		// printf("Randomi=%i Randomj=%i (%i,%i)\n",Randomi,Randomj,i,j);
		// printf("Person[%i][%i].Isolation=%i\n",Randomi,Randomj,Person[Randomi][Randomj].Isolation);

		if (Person[Randomi][Randomj].Isolation == IsolationNo && (Person[Randomi][Randomj].Health != H || Person[Randomi][Randomj].Health != ICU))
		{
			// printf("Isolou\n");
			Person[Randomi][Randomj].Isolation = IsolationYes;
			mute++;
		}
	} while (mute < MaximumIsolated);

	printf("Maximum isolated is %i\n", mute);
}
