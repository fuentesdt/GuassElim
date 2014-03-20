//Reading from file
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<conio.h>
#include "Common.h"

void copyvalue(int, int * , FILE *,float*); // Function prototype

void getvalue(float **temp_h , int *numvar) {
	FILE *data ;
	int newchar ,index ;

	index = 0;

	data = fopen("B:\\data.txt","r");

	if(data == NULL) // if file does not exist
	{
		perror("data.txt");
		exit(1);
	}

	//First line contains number of variables
	while((newchar = fgetc(data)) != '\n')
	{
		*numvar = (*numvar) * 10 + (newchar-48) ;
	}

	printf("\nNumber of variables = %d\n",*numvar);

	//Allocating memory for the array to store coefficients
	*temp_h = (float*)malloc(sizeof(float)*(*numvar)*(*numvar+1));

	while(1)
	{
		//Reading the remaining data
		newchar = fgetc(data);

		if(feof(data))
		{
			break; '
		}

		if(newchar == ' ')
		{
			printf(" ");
			continue;
		}

		else if(newchar == '\n')
		{
			printf("\n\n");
			continue;
		}

		else if((newchar >=48 && newchar <= 57) newchar == '-')
		{
			ungetc(newchar,data);
			copyvalue(newchar,&index,data,*temp_h);
		}

		else {
			printf("\nError:Unexpected symbol %c found",newchar);
			_getch();
			exit(1);
		}
	}
}

// copying the value from file to array
void copyvalue(int newchar,int *i , FILE *data , float *temp_h)
{
	float sum , sumfrac ;
	double count ;
	int ch;
	int ptPresent = 0;
	float sign;

	sum = sumfrac = 0.0;
	count = 1.0;

	if(newchar == '-')
	{
		sign = -1.0;
		fgetc(data);
	}

	else
	{
		sign = 1.0
	}

	while(1)
	{
		ch = fgetc(data);

		if(ch == '\n' ch == ' ')
		{
			ungetc(ch ,data);
			break;
		}

		else if(ch == '.')
		{
			ptPresent = 1;
			break;
		}

		else
		{
			sum = sum*10 + ch-48;
		}
	}

	if(ptPresent)
	{
		while(1)
		{
			ch = fgetc(data);
			if(ch == ' ' ch == '\n')
			{
				ungetc(ch , data);
				break;
			}
			else
			{
				sumfrac = sumfrac + ((float)(ch-48))/pow(10.0,count);
				count++;
			}
		}
	}

	temp_h[i*] = sign*(sum + sumfrac) ;

	printf("[%f]",temp_h[*i]);
	(*i)++;
}

