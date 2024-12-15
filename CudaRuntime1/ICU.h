void ICUfunc(int i, int j)
{
	Person[i][j].TimeOnState++;

	if (Person[i][j].Days >= Person[i][j].AgeDeathDays) // morte natural
	{
		Person[i][j].Swap = Dead;

		New_Dead++;
	}
	else //did not die (natural death)
	{
		if (Person[i][j].TimeOnState >= Person[i][j].StateTime) // time in H (IS severe) in hospital
		{
			aleat();
			if (rn < ProbRecoveryICU[Person[i][j].AgeYears])
			{
				Person[i][j].Swap = Recovered;

				AvailableBedsICU++;

				New_Recovered++;
			}
			else
			{
				Person[i][j].Swap = DeadCovid;

				AvailableBedsICU++;

				New_DeadCovid++;
			}
		}
		else
			Person[i][j].Swap = ICU;
	}         // did not die (natural death)
}            // end complete function