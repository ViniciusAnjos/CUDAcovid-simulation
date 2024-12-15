void Efunc(int i, int j)
{
	Person[i][j].TimeOnState++;

	if (Person[i][j].Days >= Person[i][j].AgeDeathDays) // natutal death
	{
		Person[i][j].Swap = Dead;

		New_Dead++;
	}
	else //did not die (natural death)
	{
		if (Person[i][j].TimeOnState >= Person[i][j].StateTime)  // time in E state is over
		{
			Person[i][j].Swap = IP;

			aleat();
			Person[i][j].StateTime = rn * (MaxIP - MinIP) + MinIP; //time in state IP

			New_IP++;
		}  // if to become I
		else
			Person[i][j].Swap = E;
	} //did not die (natural death)
}





































