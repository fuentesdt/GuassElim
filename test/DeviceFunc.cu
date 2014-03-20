//Assigning memory on device and defining Thread Block size
//Call to the Kernel(function) that will run on the GPU

#include<cuda.h>
#include<stdio.h>
#include "Common.h"

__device__ __global__ void Kernel(float *, float * , int);

void DeviceFunc(float *temp_h , int numvar , float *temp1_h)
{
	float *a_d , *b_d;

	//Memory allocation on the device
	cudaMalloc(&a_d,sizeof(float)*(numvar)*(numvar+1));
	cudaMalloc(&b_d,sizeof(float)*(numvar)*(numvar+1));

	//Copying data to device from host
	cudaMemcpy(a_d, temp_h, sizeof(float)*numvar*(numvar+1),cudaMemcpyHostTo Device);

	//Defining size of Thread Block
	dim3 dimBlock(numvar+1,numvar,1);
	dim3 dimGrid(1,1,1);

	//Kernel call
	Kernel<<<dimGrid , dimBlock>>>(a_d , b_d , numvar);

	//Copying data to host from device
	cudaMemcpy(temp1_h,b_d,sizeof(float)*numvar*(numvar+1),cudaMemcpyDeviceT oHost);

	//Deallocating memory on the device
	cudaFree(a_d);
	cudaFree(b_d);
}

