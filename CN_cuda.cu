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

void InitHostMem(double *Layer1_Weights_CPU, double *Layer2_Weights_CPU, double *Layer3_Weights_CPU, double *Layer4_Weights_CPU, double *Layer5_Weights_CPU)
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
	//Layer 3 Weights
	FILE * pFile3 = fopen ("data/conv3.txt","rb");
	if (pFile3 != NULL)
	{
		printf("File 3 Opened\n");
		char s[6000000] = "";
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
			Layer3_Weights_CPU[i] = temp_num;
			i++;
			index++;
			if(i==51200)
			{
				printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//printf("Last Value: %.8f\n",Layer3_Weights_CPU[51100]);
		fclose (pFile3);
	}	
	
	if (!pFile3)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	//Layer 4 Weights
	FILE * pFile4 = fopen ("data/ip1.txt","rb");
	if (pFile4 != NULL)
	{
		printf("File 4 Opened\n");
		char s[8000000] = "";
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
			Layer4_Weights_CPU[i] = temp_num;
			i++;
			index++;
			if(i==65536)
			{
				printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//printf("First Value: %.8f\n",Layer4_Weights_CPU[0]);
		fclose (pFile4);
	}	
	
	if (!pFile4)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	//Layer 5 Weights
	FILE * pFile5 = fopen ("data/ip2.txt","rb");
	if (pFile5 != NULL)
	{
		printf("File 5 Opened\n");
		char s[80000] = "";
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
			Layer5_Weights_CPU[i] = temp_num;
			i++;
			index++;
			if(i==576)
			{
				printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//printf("Last Value: %.8f\n",Layer5_Weights_CPU[575]);
		fclose (pFile5);
	}	
	
	if (!pFile5)
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
__global__ void ExecuteFirstLayer(double *Layer1_Weights_CPU, int *Data_Layer_CPU_R, int *Data_Layer_CPU_G, int *Data_Layer_CPU_B, double *Layer1_Features)
{
	//printf("First Layer Execution\n");
	int tid = threadIdx.x + threadIdx.y*32;
	int x = threadIdx.x;
	int y = threadIdx.y;
	for(int f=0; f<32; f++)
	{
				double result = 0;
				for(int i = x-2; i<=x+2; i++)
				{
    					for(int j=y-2; j<=y+2; j++)
    					{
						int x_index = i-x+2;
						int y_index = j-y+2;
						int m = (y_index)+(x_index)*5;
         					if(i<0 || j<0)
						{
							result+= 0;						
						}
         					else if(j>31 || i>31)
						{
							result+= 0;
						}
         					else
						{
							result += Data_Layer_CPU_R[(y_index-2) + x*32 + y + (x_index-2)*32]*Layer1_Weights_CPU[m+f*75] + Data_Layer_CPU_G[(y_index-2) + x*32 + y + (x_index-2)*32]*Layer1_Weights_CPU[m+25+f*75] + Data_Layer_CPU_B[(y_index-2) + x*32 + y + (x_index-2)*32]*Layer1_Weights_CPU[m+50+f*75];			
						}
					}
				} 
				Layer1_Features[f*32*32+x*32+y] = result;
	}
}

__global__ void ExecuteSecondLayer(double *Layer2_Weights_CPU, double *Layer2_Features, double *Layer2_pool_GPU)
{
	//printf("Second Layer Execution\n");
	double Features = 0;
	int x = threadIdx.x;
	int y = threadIdx.y;
	for(int f=0; f<32; f++)
	{
		Features = 0;
		//double mask[32][25];
		//double input[32][25];
		//double Features[32][16][16];
		/*for(int n=0; n<32; n++)
		{
			for(int i=0; i<25; i++)
			{
				mask[n][i] = Layer2_Weights_CPU[i+f*25*32+n*25];
				//printf("%.8f ", mask[n][i]);
			}
			//printf("\n");
		}*/
		//printf("Weights Load Complete\n");
		for(int n=0; n<32; n++)
		{
			if(x<16)//for(int x=0; x<16; x++)
			{
				if(y<16)//for(int y=0; y<16; y++)
				{
					double result = 0;
					for(int i = x-2; i<=x+2; i++)
					{
    						for(int j=y-2; j<=y+2; j++)
    						{
							int x_index = i-x+2;
							int y_index = j-y+2;
							int m = (y_index)+(x_index)*5;
         						if(i<0 || j<0)
							{
             					 		//input[n][(y_index)+(x_index)*5] = 0;
								result+=0;
							}
         						else if(j>15 || i>15)
							{
              							//input[n][(y_index)+(x_index)*5] = 0;
								result+=0;	
							}
         						else
							{
               							result+= Layer2_pool_GPU[n*16*16 + (x_index+x-2)*16 + (y_index+y-2)]*Layer2_Weights_CPU[m+f*25*32+n*25];			
							}
						}
					}
					/*for(int i=0; i<25; i++)
					{
						result+= input[n][i]*mask[n][i]; 
						//printf("%.8f ",input[n][i]);
					} */  
					Features += result;
					//printf("%f [%d][%d][%d]\n", result,n,x,y);
				}
			}
		}
		if(Features<0)
			Features = 0;
		Layer2_Features[f*16*16 + x*16 + y] = Features;
		/*if((x==0) && (y==0))
			printf("%.8f\n",Features);*/
	}
}

__global__ void ExecuteThirdLayer(double *Layer3_Weights_CPU, double *Layer3_Features, double *Layer3_pool_GPU)
{
	//printf("Third Layer Execution\n");
	double Features = 0;
	int x = threadIdx.x;
	int y = threadIdx.y;
	for(int f=0; f<64; f++)
	{
		Features = 0;
		/*double mask[32][25];
		double input[32][25];
		double Features[64][8][8];
		for(int n=0; n<32; n++)
		{
			for(int i=0; i<25; i++)
			{
				mask[n][i] = Layer3_Weights_CPU[i+f*25*32+n*25];
				//printf("%.8f ", mask[n][i]);
			}
			//printf("\n");
		}*/
		//printf("Weights Load Complete\n");
		for(int n=0; n<32; n++)
		{
			if(x<8)//for(int x=0; x<8; x++)
			{
				if(y<8)//for(int y=0; y<8; y++)
				{
					double result = 0;
					for(int i = x-2; i<=x+2; i++)
					{
    						for(int j=y-2; j<=y+2; j++)
    						{
							int x_index = i-x+2;
							int y_index = j-y+2;
							int m = (y_index)+(x_index)*5;
         						if(i<0 || j<0)
							{
             					 		//input[n][(y_index)+(x_index)*5] = 0;
								result+=0;
							}
         						else if(j>7 || i>7)
							{
              							//input[n][(y_index)+(x_index)*5] = 0;
								result+=0;
							}
         						else
							{
               							result+= Layer3_pool_GPU[n*8*8 + (x_index+x-2)*8 + (y_index+y-2)]*Layer3_Weights_CPU[m+f*25*32+n*25];			
							}
						}
					}
					//double result = 0;
					/*for(int i=0; i<25; i++)
					{
						result+= input[n][i]*mask[n][i]; 
						//printf("%.8f ",input[n][i]);
					} */  
					Features += result;
					//printf("%f [%d][%d][%d]\n", result,n,x,y);
				}
			}
		}
		if(Features<0)
			Features = 0;
		Layer3_Features[f*8*8 + x*8 + y] = Features;
		//if((x==0) && (y==0))
			//printf("%.8f\n",Features);
		/*for(int n=0; n<32; n++)
		{
			for(int x=0; x<8; x++)
			{
				for(int y=0; y<8; y++)
				{
					Layer3_Features[f][x][y]+= Features[n][x][y];
				}
			}
		}*/
	}
	/*for(int f=0; f<64; f++)
	{
		for(int x=0; x<8; x++)
		{
			for(int y=0; y<8; y++)
			{
				if(Layer3_Features[f][x][y] < 0)
					Layer3_Features[f][x][y] = 0;
			}	
		}
		//printf("\n");
	}*/
	//printf("First Value: %.8f\n",Layer3_Features[63][4][0]);
}

__global__ void ExecuteFourthLayer(double *Layer4_Weights_CPU, double *Layer4_Features, double *Pool3_Layer_Features)
{
	//printf("Fourth Layer Execution\n");
	int n = threadIdx.x;
	//for(int n=0;n<64; n++)
	{
		double result = 0;
		for(int f=0; f<64; f++)
		{
			for(int x=0; x<4; x++)
			{
				for(int y=0; y<4; y++)
				{
					result+= Pool3_Layer_Features[f*4*4 +x*4 + y] * Layer4_Weights_CPU[y+(x*4)+(f*4*4)+(n*4*4*64)];
				}
			}
		}
		Layer4_Features[n] = result;
		//printf("%.8f ",result);
		//result = 0;
	}
	//printf("\n");
	//if(n==0)
		//printf("%.8f", Layer4_Features[n]);
}

__global__ void ExecuteFifthLayer(double *Layer5_Weights_CPU, double *Layer5_Features, double *Layer4_Features)
{
	//printf("Fifth Layer Execution\n");
	int n = threadIdx.x;
	if(n<9)//for(int n=0;n<9; n++)
	{
		double result = 0;
		for(int f=0; f<64; f++)
		{
			result+= Layer4_Features[f] * Layer5_Weights_CPU[f+n*64];
		}
		Layer5_Features[n] = result;
		printf("%.8f ",result);
		result = 0;
	}
	//printf("\n");
	//if(n==0)
		//printf("%.8f", Layer5_Features[n]);
}

__global__ void pooling1(double *Layer2_Neurons_GPU,double *Layer2_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc)
{
    //printf("pooling Activation layer \n");
    int row = threadIdx.x;
    int col = threadIdx.y;
    double max = 0.0;
    {
        for(int output =0;output < out ;output++)
        {
            if(row%2 != 0)//for(int row =1; row <= 31 ;row+=2)
            { 
                if(col%2 != 0)//for(int col =1; col <= 31 ;col+=2)
                {
                    for(int i = row-1; i <= row+1; i++)
                    {   
			if(i>31) break;        
                        for(int j = col-1; j <= col+1; j++)
                        {
			    if(j>31) break;
                            if(max < ((Layer2_Neurons_GPU[output*32*32+i*32+j])))
                                max =   ((Layer2_Neurons_GPU[output*32*32+i*32+j])) ;

                        }
                    }
		    if(max<0)
			max = 0;
                    Layer2_pool_GPU[output*16*16+(row-1)*8+(col-1)/2] = max;
                    //printf("%f %d \n",max, (((row-1)*8)+((col-1)/2) + output*16*16));
		        /*if(row == 1 && col == 1)
    			{
			     printf("%.8f\n",max);
    			}  */   
                    max = 0.0;   
                }
            }
	//printf("\n");
        }
    }
    /*if(row == 0 && col == 0)
    {
	printf("%.8f\n",max);
    }*/
}

__global__ void pooling2(double *Layer2_Neurons_GPU,double *Layer2_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc)
{
    //printf("pooling 2 layer \n");
    double avg = 0.0;
    int count = 0;
    int row = threadIdx.x;
    int col = threadIdx.y;
    {
        for(int output =0;output < out ;output++)
        {
            if((row%2 != 0) && (row<16))//for(int row =1; row <= 16 ;row+=2)
            { 
                if((col%2 != 0) && (col<16))//for(int col =1; col <= 16 ;col+=2)
                {
                    for(int i = row-1; i <= row+1; i++)
                    {   
			if(i>15) break;        
                        for(int j = col-1; j <= col+1; j++)
                        {
			    if(j>15) break;
                            avg+= Layer2_Neurons_GPU[output*16*16 + i*16 + j];
			    count = count + 1;

                        }
                    }
                    Layer2_pool_GPU[output*8*8+(row-1)*4+(col-1)/2] = avg/count;
                    //printf("%f %d \n",max, (((row-1)*8)+((col-1)/2) + output*16*16));     
                    avg = 0.0;   
		    count=0;
                }
            }
	//printf("\n");
        }
    }
    //for(int i=0; i<8; i++)
    	//printf("%.8f ",Layer2_pool_GPU[31][7][i]);
    //printf("\n");
}

__global__ void pooling3(double *Layer3_Neurons_GPU,double *Layer3_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc)
{
    //printf("pooling 3 layer \n");
    double avg = 0.0;
    int count = 0;
    int row = threadIdx.x;
    int col = threadIdx.y;
    {
        for(int output =0;output < out ;output++)
        {
            if((row%2 != 0) && (row<8))//for(int row =1; row <= 8 ;row+=2)
            { 
                if((col%2 != 0) && (col<8))//for(int col =1; col <= 8 ;col+=2)
                {
                    for(int i = row-1; i <= row+1; i++)
                    {   
			if(i>7) break;        
                        for(int j = col-1; j <= col+1; j++)
                        {
			    if(j>7) break;
                            avg+= ((Layer3_Neurons_GPU[output*8*8 + i*8 + j]));
			    count++;

                        }
                    }
                    Layer3_pool_GPU[output*4*4+(row-1)*2+(col-1)/2] = avg/count;
                    //printf("%f %d \n",max, (((row-1)*8)+((col-1)/2) + output*16*16));     
                    avg = 0.0;   
		    count=0;
                }
            }
	//printf("\n");
        }
    }
    /*for(int i=0; i<4; i++)
    	printf("%.8f ",Layer3_pool_GPU[63][3][i]);
    printf("\n");*/
    /*if(row == 0 && col == 0)
    {
	printf("%.8f\n",Layer3_pool_GPU[0]);
    }*/
}

void NeuralNetwork()
{
	cudaError_t err;
	int deviceCount;                                                         
	cudaGetDeviceCount(&deviceCount);                
	if (deviceCount == 0) {                                                  
		fprintf(stderr, "There is no device.\n");                            
		exit(EXIT_FAILURE);                                                  
	}                                                                        
	int dev;                                                                 
	for (dev = 0; dev < deviceCount; ++dev) {                                
		cudaDeviceProp deviceProp;                                           
		cudaGetDeviceProperties(&deviceProp, dev);   
		if (deviceProp.major >= 1)                                           
			break;                                                           
	}                                                                        
	if (dev == deviceCount) {                                                
		fprintf(stderr, "There is no device supporting CUDA.\n");            
		exit(EXIT_FAILURE);                                                  
	}                                                                        
	else                                                                     
		cudaSetDevice(dev);
	//printf("Started");
	double *Layer1_Weights_CPU = (double*) malloc (3*32*32* NUM * sizeof(double));
	double *Layer2_Weights_CPU = (double*) malloc (5*5*32*32* NUM * sizeof(double));
	double *Layer3_Weights_CPU = (double*) malloc (5*5*32*64* NUM * sizeof(double));
	double *Layer4_Weights_CPU = (double*) malloc (64*4*4*64* NUM * sizeof(double));
	double *Layer5_Weights_CPU = (double*) malloc (64*9* NUM * sizeof(double));
	int *Data_Layer_CPU_R = (int*) malloc (32*32*NUM*sizeof(int));
	int *Data_Layer_CPU_G = (int*) malloc (32*32*NUM*sizeof(int));
	int *Data_Layer_CPU_B = (int*) malloc (32*32*NUM*sizeof(int));
	int *Data_Layer_GPU_R = (int*) malloc (32*32*NUM*sizeof(int));
	int *Data_Layer_GPU_G = (int*) malloc (32*32*NUM*sizeof(int));
	int *Data_Layer_GPU_B = (int*) malloc (32*32*NUM*sizeof(int));
	int *Data_Layer_CPU = (int*) malloc (3*32*32*NUM*sizeof(int));
	InitHostMem(Layer1_Weights_CPU, Layer2_Weights_CPU, Layer3_Weights_CPU, Layer4_Weights_CPU, Layer5_Weights_CPU);
	LoadInput(Data_Layer_CPU);
	ConvertInput(Data_Layer_CPU_R, Data_Layer_CPU_G, Data_Layer_CPU_B, Data_Layer_CPU);
	double *Layer1_Features;
	double *Layer1_Weights_GPU;
	err = cudaMalloc((void**) &Layer1_Features, 32*32*32* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &Layer1_Weights_GPU, 2400* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &Data_Layer_GPU_R, 32*32* NUM * sizeof(int));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &Data_Layer_GPU_G, 32*32* NUM * sizeof(int));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &Data_Layer_GPU_B, 32*32* NUM * sizeof(int));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	printf("Malloc completed\n");
	cudaMemcpy(Layer1_Weights_GPU,Layer1_Weights_CPU, sizeof(double)*2400*NUM, cudaMemcpyHostToDevice);
	cudaMemcpy(Data_Layer_GPU_R,Data_Layer_CPU_R, 32*32* NUM * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Data_Layer_GPU_G,Data_Layer_CPU_G, 32*32* NUM * sizeof(int), cudaMemcpyHostToDevice);
	err = cudaMemcpy(Data_Layer_GPU_B,Data_Layer_CPU_B, 32*32* NUM * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	printf("Memcpy completed\n");
	dim3 n_threads(32,32,1);
	dim3 n_blocks(1,1,1); 
	cudaThreadSynchronize();
	ExecuteFirstLayer<<<n_blocks,n_threads>>>(Layer1_Weights_GPU, Data_Layer_GPU_R, Data_Layer_GPU_G, Data_Layer_GPU_B, Layer1_Features);
	
	err = cudaGetLastError();
        if (err != cudaSuccess)
        {
        	fprintf(stderr, "1st LayerKernel execution failed (error code %s)\n", cudaGetErrorString(err));
       		exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	double *Pool_Layer_Features;
	err = cudaMalloc((void**) &Pool_Layer_Features, 32*16*16* NUM * sizeof(double));
	pooling1<<<n_blocks,n_threads>>>(Layer1_Features, Pool_Layer_Features, 32, 16, 16, 5, 2, 32, 32);
	cudaThreadSynchronize();
	err = cudaGetLastError();
        if (err != cudaSuccess)
        {
        	fprintf(stderr, "1st Pool Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
       		exit(EXIT_FAILURE);
        }
	double *Layer2_Weights_GPU;
	err = cudaMalloc((void**) &Layer2_Weights_GPU, 5*5*32*32* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaMemcpy(Layer2_Weights_GPU,Layer2_Weights_CPU, sizeof(double)*5*5*32*32*NUM, cudaMemcpyHostToDevice);
	double *Layer2_Features;	
	err = cudaMalloc((void**) &Layer2_Features, 32*16*16* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	ExecuteSecondLayer<<<n_blocks,n_threads>>>(Layer2_Weights_GPU, Layer2_Features, Pool_Layer_Features);
	cudaThreadSynchronize();
	err = cudaGetLastError();
        if (err != cudaSuccess)
        {
        	fprintf(stderr, "2nd Layer Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
       		exit(EXIT_FAILURE);
        }
	double *Pool2_Layer_Features;
	cudaMalloc((void**) &Pool2_Layer_Features, 32*8*8* NUM * sizeof(double));	
	pooling2<<<n_blocks,n_threads>>>(Layer2_Features, Pool2_Layer_Features, 32, 8, 8, 5, 2, 16, 16);
	cudaThreadSynchronize();
	err = cudaGetLastError();
        if (err != cudaSuccess)
        {
        	fprintf(stderr, "2nd Pool Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
       		exit(EXIT_FAILURE);
        }
	double *Layer3_Weights_GPU;
	err = cudaMalloc((void**) &Layer3_Weights_GPU, 5*5*32*64* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaMemcpy(Layer3_Weights_GPU,Layer3_Weights_CPU, sizeof(double)*5*5*32*64*NUM, cudaMemcpyHostToDevice);
	double *Layer3_Features;	
	err = cudaMalloc((void**) &Layer3_Features, 64*8*8* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	ExecuteThirdLayer<<<n_blocks,n_threads>>>(Layer3_Weights_GPU, Layer3_Features, Pool2_Layer_Features);
	cudaThreadSynchronize();
	err = cudaGetLastError();
        if (err != cudaSuccess)
        {
        	fprintf(stderr, "3rd Layer Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
       		exit(EXIT_FAILURE);
        }
	double *Pool3_Layer_Features;
	cudaMalloc((void**) &Pool3_Layer_Features, 64*4*4* NUM * sizeof(double));
	pooling3<<<n_blocks,n_threads>>>(Layer3_Features, Pool3_Layer_Features, 64, 4, 4, 5, 2, 8, 8);
	cudaThreadSynchronize();
	err = cudaGetLastError();
        if (err != cudaSuccess)
        {
        	fprintf(stderr, "3rd Pool Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
       		exit(EXIT_FAILURE);
        }
	double *Layer4_Features;
	cudaMalloc((void**) &Layer4_Features, 64*sizeof(double));
	double *Layer4_Weights_GPU;
	err = cudaMalloc((void**) &Layer4_Weights_GPU, 64*4*4*64* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaMemcpy(Layer4_Weights_GPU,Layer4_Weights_CPU, sizeof(double)*64*4*4*64*NUM, cudaMemcpyHostToDevice);
	ExecuteFourthLayer<<<1,64>>>(Layer4_Weights_GPU, Layer4_Features, Pool3_Layer_Features);	
	cudaThreadSynchronize();
	err = cudaGetLastError();
        if (err != cudaSuccess)
        {
        	fprintf(stderr, "4th Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
       		exit(EXIT_FAILURE);
        }
	double *Layer5_Features;
	cudaMalloc((void**) &Layer5_Features, 9*sizeof(double));
	double *Layer5_Weights_GPU;
	err = cudaMalloc((void**) &Layer5_Weights_GPU, 64*9* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaMemcpy(Layer5_Weights_GPU,Layer5_Weights_CPU, sizeof(double)*64*9*NUM, cudaMemcpyHostToDevice);
	ExecuteFifthLayer<<<1,32>>>(Layer5_Weights_GPU, Layer5_Features, Layer4_Features);	
	cudaThreadSynchronize();
	err = cudaGetLastError();
        if (err != cudaSuccess)
        {
        	fprintf(stderr, "5th Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
       		exit(EXIT_FAILURE);
        }

}


