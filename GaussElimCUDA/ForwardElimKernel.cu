// Kernel for forward elimination in Gauss elimination

#include <cuda.h>

__global__ void ForwardElimKernel(double d_A[3][4], double d_piv[3], nDim)
{
	// Assign matrix elements to blocks and threads
	int i = blockDim.y*blockIdx.y + threadIdx.y;
	int j = blockDim.x*blockIdx.x + threadIdx.x;

	// Parallel forward elimination
	for (int k = 0; k <= nDim-2; k++)
	{
		d_piv[i] = d_A[i][k]/d_A[k][k];
		__syncthreads();
		if (((i>k) && (i<nDim)) && ((j>=k) && (j<=nDim)))
			d_A[i][j] -= d_piv[i]*d_A[k][j];
		__syncthreads();
	}
}
