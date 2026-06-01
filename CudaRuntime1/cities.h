
void cities(const int city)
{

    switch (city)
    {
    case SP:

        //S�o paulo
        Density = HIGH;
        BEDSPOP = 0.00247452;
        ICUPOP = 0.00043782;
        MaxRandomContacts = 18.5;   // doc: SP 2-19 (convencao -0.5, igual Rocinha)
        MinRandomContacts = 1.5;


        break;

    case ROC:

        //Rocinha
        Density = HIGH;
        BEDSPOP = 0.00055111;
        ICUPOP = 0.00014592;
        MaxRandomContacts = 119.5;
        MinRandomContacts = 1.5;

        break;

    case BRA:

        //Bras�lia
        Density = LOW;
        BEDSPOP = 0.00260879;
        ICUPOP = 0.00040114;
        MaxRandomContacts = 2.5;
        MinRandomContacts = 1.5;

        break;

    case MAN:

        //Manaus
        Density = LOW;
        BEDSPOP = 0.00187124;
        ICUPOP = 0.00027858;
        MaxRandomContacts = 2.5;
        MinRandomContacts = 1.5;



        break;


    default:
        printf("City not found\n");


    }

    NumberOfHospitalBeds = BEDSPOP * N;
    NumberOfICUBeds = ICUPOP * N;

}

