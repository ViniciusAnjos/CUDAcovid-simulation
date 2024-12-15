void IPfunc(int i, int j)
{
	Person[i][j].TimeOnState++;


	if (Person[i][j].Days >= Person[i][j].AgeDeathDays) // morte natural
	{
		Person[i][j].Swap = Dead;

		New_Dead++;
	}
	else //did not die (natural death)
	{
		if (Person[i][j].TimeOnState >= Person[i][j].StateTime) // time in IP state is over
		{
			aleat();
			if (rn < ProbIPtoIA) // move to IA state
			{

				Person[i][j].Swap = IA;

				aleat();
				Person[i][j].StateTime = rn * (MaxIA - MinIA) + MinIA; //time in state IP

				New_IA++;
			}
			else  // move to some type of IS
			{

				aleat();
				if (rn < ProbToBecomeISLight)
				{

					Person[i][j].Swap = ISLight;

					aleat();
					Person[i][j].StateTime = rn * (MaxISLight - MinISLight) + MinISLight; //time in state IP

					New_ISLight++;
				}
				else if ((rn >= ProbToBecomeISLight) && (rn <= (ProbToBecomeISLight + ProbToBecomeISModerate)))
				{


					Person[i][j].Swap = ISModerate;

					aleat();
					Person[i][j].StateTime = rn * (MaxISModerate - MinISModerate) + MinISModerate; //time in state ISModerate

					New_ISModerate++;
				}
				else
				{

					Person[i][j].Swap = ISSevere;

					aleat();
					Person[i][j].StateTime = rn * (MaxISSevere - MinISSevere) + MinISSevere; //time in state IP

					New_ISSevere++;
				}
			}  // end IS any type
		} // end time on state
		else
		{
			Neighborsinfectedfunc(i, j);

			Person[i][j].Swap = IP;
		}
	} // do not die (natural death)
}	       	// end of function


