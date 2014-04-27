
/*
 * Device code
 */

__global__ void ParallelGaussElim(
	int const nDim_image,
	int const nDim_matrix,
	double* d_A,
	double* d_b,
	double* d_x)
{
	// Assign image pixels to blocks and threads
	int i_image = blockDim.x*blockIdx.x + threadIdx.x;
	//int i_image = blockDim.y*blockIdx.y + threadIdx.y;
//printf("blockDim.x = %i \n",blockDim.x);
//printf("blockIdx.x = %i \n",blockIdx.x);
//printf("threadIdx.x = %i \n",threadIdx.x);
//printf("i_image = %i \n", i_image);
	//int offset = (j_image + i_image*nDim_image)*nDim_matrix*nDim_matrix;
	int offset_2d = i_image*nDim_matrix*nDim_matrix;
	int offset_1d = i_image*nDim_matrix;
//printf("offset = %i \n", offset);
	// Gauss elimination
	//int nDim_local = 8;
	//double local_A[nDim_local*nDim_local] = 0;
	for (int k=0; k<nDim_matrix-1; k++)
	{
		for (int i=k+1;	i<nDim_matrix; i++)
		{
			double pivot = d_A[offset_2d+i+k*nDim_matrix]/d_A[offset_2d+k+k*nDim_matrix];
			for (int j=k; j<nDim_matrix; j++)
			{
				d_A[offset_2d+i+j*nDim_matrix] -= pivot*d_A[offset_2d+k+j*nDim_matrix];
			}
			d_b[offset_1d+i] -= pivot*d_b[offset_1d+k];
		}
	}

	// Backward substitution
	for (int i=nDim_matrix-1; i>=0; i--)
	{
		d_x[offset_1d+i] = d_b[offset_1d+i];
		for (int j=nDim_matrix-1; j>i; j--)
		{
			d_x[offset_1d+i] -= d_A[offset_2d+i+j*nDim_matrix]*d_x[offset_1d+j];
		}
        	d_x[offset_1d+i] = d_x[offset_1d+i]/d_A[offset_2d+i+i*nDim_matrix];
if (d_x[offset_1d+i] /= 1) printf ("blkdim,id,tdid = %i,%i,%i \n",blockDim.x,blockIdx.x,threadIdx.x);
	}
}
