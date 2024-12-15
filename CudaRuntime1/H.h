void Hfunc(int i, int j)
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
			if (rn < ProbRecoveryH[Person[i][j].AgeYears])
			{
				Person[i][j].Swap = Recovered;

				AvailableBeds++;

				New_Recovered++;
			}
			else
			{
				if (AvailableBedsICU > 0)
				{
					Person[i][j].Swap = ICU;

					AvailableBedsICU--;

					AvailableBeds++;

					aleat();
					Person[i][j].StateTime = rn * (MaxICU - MinICU) + MinICU; //time in state ICU

					New_ICU++;
				}
				else   // no beds available
				{
					Person[i][j].Swap = DeadCovid;

					AvailableBeds++;

					New_DeadCovid++;
				}
			} // not recovered
		}
		else
			Person[i][j].Swap = H;
	}         // did not die (natural death)
}            // end complete function