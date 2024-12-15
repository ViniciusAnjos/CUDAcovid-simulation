void ISfunc(int i, int j)
{
	Person[i][j].TimeOnState++;


	if (Person[i][j].Days >= Person[i][j].AgeDeathDays) // morte natural
	{
		Person[i][j].Swap = Dead;

		New_Dead++;
	}
	else //did not die (natural death)
	{
		if (Person[i][j].Health == IA)
		{
			if (Person[i][j].TimeOnState >= Person[i][j].StateTime) // time in IA state is over
			{
				Person[i][j].Swap = Recovered;

				New_Recovered++;
			}
			else
			{
				Neighborsinfectedfunc(i, j);

				Person[i][j].Swap = IA;
			}
		}                                                   // end if (IA)
		else if (Person[i][j].Health == ISLight)
		{
			if (Person[i][j].TimeOnState >= Person[i][j].StateTime) // time in IP state is over
			{
				aleat();
				if (rn < ProbISLightToISModerate)
				{
					Person[i][j].Swap = ISModerate;

					aleat();
					Person[i][j].StateTime = rn * (MaxISModerate - MinISModerate) + MinISModerate; //time in state ISModerate

					New_ISModerate++;
				}
				else
				{
					Person[i][j].Swap = Recovered;

					New_Recovered++;
				}
			}
			else
			{
				Neighborsinfectedfunc(i, j);

				Person[i][j].Swap = ISLight;
			}
		}                                                // end if (ISLight)
		else if (Person[i][j].Health == ISModerate)
		{
			if (Person[i][j].TimeOnState >= Person[i][j].StateTime) // time in IP state is over
			{
				aleat();
				if (rn < ProbRecoveryModerate[Person[i][j].AgeYears])
				{
					Person[i][j].Swap = Recovered;

					New_Recovered++;
				}
				else
				{
					Person[i][j].Swap = ISSevere;

					aleat();
					Person[i][j].StateTime = rn * (MaxISSevere - MinISSevere) + MinISSevere; //time in state IP

					New_ISSevere++;
				}
			}
			else
				Person[i][j].Swap = ISModerate;
		}                                                 // end if (ISModerate)
		else if (Person[i][j].Health == ISSevere)
		{
			if (Person[i][j].TimeOnState >= Person[i][j].StateTime) // time in IS (severe) state is over
			{
				aleat();
				if (rn < ProbRecoverySevere[Person[i][j].AgeYears])
				{
					Person[i][j].Swap = Recovered;

					New_Recovered++;
				}
				else
				{
					if (AvailableBeds > 0)
					{
						AvailableBeds--;

						Person[i][j].Swap = H;

						aleat();
						Person[i][j].StateTime = rn * (MaxH - MinH) + MinH; //time in hospital	

						New_H++;
					}
					else   // no beds available
					{
						Person[i][j].Swap = DeadCovid;

						New_DeadCovid++;
					}
				} // not recovered
			}
			else // time in IS (severe) state is running
				Person[i][j].Swap = ISSevere;
		}      // end if (ISSevere)
	}         // did not die (natural death)
}            // end complete function