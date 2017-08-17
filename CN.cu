#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <assert.h>
using namespace std;

// includes, project
//#include <cutil.h>

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);
void NeuralNetwork();

unsigned g_verbose;
unsigned NUM;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
	int i, commandline_error;
	commandline_error = 0;
	g_verbose = 0;
	if (argc >= 2) {
		NUM = atoi(argv[1]);
		for (i=2; i < argc;i++) {
			if (argv[i][0] == '-') {
				switch (argv[i][1]) {
				case 'v': g_verbose = 1;
					break;
				default: commandline_error=1;
				}
			}
			else commandline_error=1;
		}
	} else commandline_error=1;

	if (commandline_error || !NUM) {
		printf("Usage: ./NN <NUM> [-v]\n");
		printf("where NUM is the number of images to process in parallel (up to 10000 for the t10k-images-idx3-ubyte database file) and -v is used to display approximately what each image looks like.\n");
		return 1;
	}


	NeuralNetwork();
    //CUT_EXIT(argc, argv);
}

void InitHostMem(double *Layer1_Weights_CPU, double *Layer2_Weights_CPU)
{
	// initial layer 1 weight
	FILE * pFile1 = fopen ("data/conv1.txt","rb");
	if (pFile1 != NULL)
	{
		printf("File Opened\n");
		char s[300000] = "";
		fread(s,sizeof(s),1,pFile1);
		printf("Done2\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			Layer1_Weights_CPU[i] = temp_num;
			i++;
			index++;
			if(i==2400)
			{
				printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile1);
	}	
	
	if (!pFile1)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	//Layer 2 Weights
	FILE * pFile2 = fopen ("data/conv2.txt","rb");
	if (pFile2 != NULL)
	{
		printf("File 2 Opened\n");
		char s[3000000] = "";
		fread(s,sizeof(s),1,pFile2);
		//printf("%s\n",s);
		printf("Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			//printf("%.8f %d\n",temp_num,i);
			Layer2_Weights_CPU[i] = temp_num;
			i++;
			index++;
			if(i==25600)
			{
				printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//printf("Last Value: %.8f\n",Layer2_Weights_CPU[25599]);
		fclose (pFile2);
	}	
	
	if (!pFile2)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
}

void LoadInput(int *Data_Layer_CPU)
{
	FILE * pFile1 = fopen ("data/speed-limit-35.txt","rb");
	if (pFile1 != NULL)
	{
		printf("File Opened\n");
		char s[300000] = "";
		fread(s,sizeof(s),1,pFile1);
		//printf("%s", s);
		printf("Done2\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		//int address = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			int temp_num = atof(temp_string);	
			Data_Layer_CPU[i] = temp_num;
			i++;
			index++;
			if(i==(32*32*3))
			{
				printf("Breaking input\n");
				break;
			}
			temp_string = strtok(NULL, delim);
			//if(temp_string != NULL)
			//	address = strlen(temp_string);
		}
		//printf("%d", Data_Layer_CPU[(32*32*3)-1]);
		fclose (pFile1);
		//printf("Last image value: %d", );
	}	
	if (!pFile1)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
}

void ConvertInput(int *Data_Layer_CPU_R, int *Data_Layer_CPU_G, int *Data_Layer_CPU_B, int *Data_Layer_CPU)
{
	for(int i=0; i<32*32*3; i+=3)
	{
		Data_Layer_CPU_R[i/3] = Data_Layer_CPU[i];
		Data_Layer_CPU_G[i/3] = Data_Layer_CPU[i+1];
		Data_Layer_CPU_B[i/3] = Data_Layer_CPU[i+2];
	}
}
void ExecuteFirstLayer(double *Layer1_Weights_CPU, int *Data_Layer_CPU_R, int *Data_Layer_CPU_G, int *Data_Layer_CPU_B, double ***Layer1_Features)
{
	for(int f=0; f<32; f++)
	{
		double maskR[25], maskG[25], maskB[25];
		int imageR[25], imageG[25], imageB[25];
		for(int i=0; i<25; i++)
		{
			maskR[i] = Layer1_Weights_CPU[i+f*75];
			maskG[i] = Layer1_Weights_CPU[i+25+f*75];
			maskB[i] = Layer1_Weights_CPU[i+50+f*75];
		}
		for(int x=0; x<32; x++)
		{
			for(int y=0; y<32; y++)
			{
				for(int i = x-2; i<=x+2; i++)
				{
    					for(int j=y-2; j<=y+2; j++)
    					{
						int x_index = i-x+2;
						int y_index = j-y+2;
         					if(i<0 || j<0)
						{
             				 		imageR[(y_index)+(x_index)*5] = 0;
							imageG[(y_index)+(x_index)*5] = 0;
							imageB[(y_index)+(x_index)*5] = 0;
						}
         					else if(j>31 || i>31)
						{
              						imageR[(y_index)+(x_index)*5] = 0;
							imageG[(y_index)+(x_index)*5] = 0;
							imageB[(y_index)+(x_index)*5] = 0;
						}
         					else
						{
               						imageR[(y_index)+(x_index)*5] = Data_Layer_CPU_R[(y_index-2) + x*32 + y + (x_index-2)*32];
               						imageG[(y_index)+(x_index)*5] = Data_Layer_CPU_G[(y_index-2) + x*32 + y + (x_index-2)*32];
               						imageB[(y_index)+(x_index)*5] = Data_Layer_CPU_B[(y_index-2) + x*32 + y + (x_index-2)*32];			
						}
					}
				}
				double result = 0;
				for(int i=0; i<25; i++)
				{
					 result+= imageR[i]*maskR[i] +imageG[i]*maskG[i] + imageB[i]*maskB[i]; 
				}   
				Layer1_Features[f][x][y] = result;
				//printf("%f ", result);
			}
		}
	}
	printf("\n");
	for(int x=0; x<32; x++)
	{
		for(int y=0; y<32; y++)
		{
			//printf("%.8f  %d\n",Layer1_Features[1][x][y], x*32+y);
		}
		//printf("\n");
	}
}

void ExecuteSecondLayer(double *Layer2_Weights_CPU, double ***Layer2_Features, double ***Layer2_pool_GPU)
{
	printf("Second Layer Executions:\n");
	for(int f=0; f<32; f++)
	{
		double mask[32][25];
		double input[32][25];
		double Features[32][16][16];
		for(int n=0; n<32; n++)
		{
			for(int i=0; i<25; i++)
			{
				mask[n][i] = Layer2_Weights_CPU[i+f*25*32+n*25];
				//printf("%.8f ", mask[n][i]);
			}
			//printf("\n");
		}
		//printf("Weights Load Complete\n");
		for(int n=0; n<32; n++)
		{
			for(int x=0; x<16; x++)
			{
				for(int y=0; y<16; y++)
				{
					for(int i = x-2; i<=x+2; i++)
					{
    						for(int j=y-2; j<=y+2; j++)
    						{
							int x_index = i-x+2;
							int y_index = j-y+2;
         						if(i<0 || j<0)
							{
             					 		input[n][(y_index)+(x_index)*5] = 0;
							}
         						else if(j>15 || i>15)
							{
              							input[n][(y_index)+(x_index)*5] = 0;
							}
         						else
							{
               							input[n][(y_index)+(x_index)*5] = Layer2_pool_GPU[n][x_index+x-2][y_index+y-2];			
							}
						}
					}
					double result = 0;
					for(int i=0; i<25; i++)
					{
						result+= input[n][i]*mask[n][i]; 
						//printf("%.8f ",input[n][i]);
					}   
					Features[n][x][y] = result;
					//printf("%f [%d][%d][%d]\n", result,n,x,y);
				}
			}
		}
		for(int n=0; n<32; n++)
		{
			for(int x=0; x<16; x++)
			{
				for(int y=0; y<16; y++)
				{
					Layer2_Features[f][x][y]+= Features[n][x][y];
				}
			}
		}
	}
	for(int f=0; f<32; f++)
	{
		for(int x=0; x<16; x++)
		{
			for(int y=0; y<16; y++)
			{
				if(Layer2_Features[f][x][y] < 0)
					Layer2_Features[f][x][y] = 0;
			}	
		}
		//printf("\n");
	}
	//printf("First Value: %.8f\n",Layer2_Features[31][15][11]);
}

void pooling1(double ***Layer2_Neurons_GPU,double ***Layer2_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc)
{
    printf("pooling Activation layer \n");
    double max = 0.0;
    {
        for(int output =0;output < out ;output++)
        {
            for(int row =1; row <= 31 ;row+=2)
            { 
                for(int col =1; col <= 31 ;col+=2)
                {
                    for(int i = row-1; i <= row+1; i++)
                    {   
			if(i>31) break;        
                        for(int j = col-1; j <= col+1; j++)
                        {
			    if(j>31) break;
                            if(max < ((Layer2_Neurons_GPU[output][i][j])))
                                max =   ((Layer2_Neurons_GPU[output][i][j])) ;

                        }
                    }
		    if(max<0)
			max = 0;
                    Layer2_pool_GPU[output][(row-1)/2][(col-1)/2] = max;
                    //printf("%f %d \n",max, (((row-1)*8)+((col-1)/2) + output*16*16));     
                    max = 0.0;   
                }
            }
	//printf("\n");
        }
    }
}

void pooling2(double ***Layer2_Neurons_GPU,double ***Layer2_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc)
{
    printf("pooling 2 layer \n");
    double avg = 0.0;
    int count = 0;
    {
        for(int output =0;output < out ;output++)
        {
            for(int row =1; row <= 16 ;row+=2)
            { 
                for(int col =1; col <= 16 ;col+=2)
                {
                    for(int i = row-1; i <= row+1; i++)
                    {   
			if(i>15) break;        
                        for(int j = col-1; j <= col+1; j++)
                        {
			    if(j>15) break;
                            avg+= ((Layer2_Neurons_GPU[output][i][j]));
			    count++;

                        }
                    }
                    Layer2_pool_GPU[output][(row-1)/2][(col-1)/2] = avg/count;
                    //printf("%f %d \n",max, (((row-1)*8)+((col-1)/2) + output*16*16));     
                    avg = 0.0;   
		    count=0;
                }
            }
	//printf("\n");
        }
    }
    for(int i=0; i<8; i++)
    	printf("%.8f ",Layer2_pool_GPU[31][7][i]);
    printf("\n");
}

void NeuralNetwork()
{
	double *Layer1_Weights_CPU = (double*) malloc (3*32*32* NUM * sizeof(double));
	double *Layer2_Weights_CPU = (double*) malloc (5*5*32*32* NUM * sizeof(double));
	int *Data_Layer_CPU_R = (int*) malloc (32*32*NUM*sizeof(int));
	int *Data_Layer_CPU_G = (int*) malloc (32*32*NUM*sizeof(int));
	int *Data_Layer_CPU_B = (int*) malloc (32*32*NUM*sizeof(int));
	int *Data_Layer_CPU = (int*) malloc (3*32*32*NUM*sizeof(int));
	InitHostMem(Layer1_Weights_CPU, Layer2_Weights_CPU);
	LoadInput(Data_Layer_CPU);
	ConvertInput(Data_Layer_CPU_R, Data_Layer_CPU_G, Data_Layer_CPU_B, Data_Layer_CPU);
	double ***Layer1_Features;
	Layer1_Features = (double***)malloc(32*sizeof(double **));
	assert(Layer1_Features!= NULL);
	for(int i=0; i<32; i++)
	{
		Layer1_Features[i] = (double**)malloc(32*sizeof(double *));
		assert(Layer1_Features[i] != NULL);
		for(int j=0; j<32; j++)
		{
			Layer1_Features[i][j] = (double*)malloc(32*sizeof(double));
		}
	}
	ExecuteFirstLayer(Layer1_Weights_CPU, Data_Layer_CPU_R, Data_Layer_CPU_G, Data_Layer_CPU_B, Layer1_Features);
	double ***Pool_Layer_Features;
	Pool_Layer_Features = (double***)malloc(32*sizeof(double **));
	assert(Pool_Layer_Features!= NULL);
	for(int i=0; i<32; i++)
	{
		Pool_Layer_Features[i] = (double**)malloc(16*sizeof(double *));
		assert(Pool_Layer_Features[i] != NULL);
		for(int j=0; j<16; j++)
		{
			Pool_Layer_Features[i][j] = (double*)malloc(16*sizeof(double));
		}
	}
	pooling1(Layer1_Features, Pool_Layer_Features, 32, 16, 16, 5, 2, 32, 32);
	double ***Layer2_Features;
	Layer2_Features = (double***)malloc(32*sizeof(double **));
	assert(Layer2_Features!= NULL);
	for(int i=0; i<32; i++)
	{
		Layer2_Features[i] = (double**)malloc(16*sizeof(double *));
		assert(Layer2_Features[i] != NULL);
		for(int j=0; j<16; j++)
		{
			Layer2_Features[i][j] = (double*)malloc(16*sizeof(double));
		}
	}
	ExecuteSecondLayer(Layer2_Weights_CPU, Layer2_Features, Pool_Layer_Features);
	double ***Pool2_Layer_Features;
	Pool2_Layer_Features = (double***)malloc(32*sizeof(double **));	
	assert(Pool2_Layer_Features!= NULL);
	for(int i=0; i<32; i++)
	{
		Pool2_Layer_Features[i] = (double**)malloc(8*sizeof(double *));
		assert(Pool2_Layer_Features[i] != NULL);
		for(int j=0; j<8; j++)
		{
			Pool2_Layer_Features[i][j] = (double*)malloc(8*sizeof(double));
		}
	}
	pooling2(Layer2_Features, Pool2_Layer_Features, 32, 8, 8, 5, 2, 16, 16);
}


