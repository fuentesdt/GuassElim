
/*
 * Device code
 */

/*
__global__ 
void GaussSolve(
         int const Nsize,
         double* d_Aug,
         double* d_Piv)
{
    for (int i=0; i<16; i++) d_Aug[i]=i;
     Assign matrix elements to blocks and threads
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    // Parallel forward elimination
    for (int k = 0; k < Nsize-1; k++)
    {
        d_Piv[i] = d_Aug[i%Nsize+k*Nsize]/d_Aug[k*(Nsize+1)];
        __syncthreads();
        if (((i%Nsize)>k) && ((i/Nsize)>=k) && ((i/Nsize)<=Nsize))
            d_Aug[i] -= d_Piv[i]*d_Aug[i-(i%Nsize)+k];
        __syncthreads();
    }

}
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

	//int offset = (j_image + i_image*nDim_image)*nDim_matrix*nDim_matrix;
	int offset = i_image*nDim_matrix*nDim_matrix;

	// Gauss elimination
	for (int k=0; k<nDim_matrix-1; k++)
	{
		for (int i=k+1;	i<nDim_matrix; i++)
		{
			double pivot = d_A[offset+i+k*nDim_matrix]/d_A[offset+k+k*nDim_matrix];
			for (int j=k; j<nDim_matrix; j++)
			{
				d_A[offset+i+j*nDim_matrix] -= pivot*d_A[offset+k+j*nDim_matrix];
			}
			d_b[offset+i] -= pivot*d_b[offset+k];
		}
	}

/*	do k=1,ndim-1
		do i=k+1,ndim
			pivot=A(i,k)/A(k,k)
			do j=k,ndim
				A(i,j)=A(i,j)-pivot*A(k,j)
			end do
			b(i)=b(i)-pivot*b(k)
		end do
	end do
*/

	// Backward substitution

	for (int i=nDim_matrix-1; i>=0; i--)
	{
		d_x[offset+i] = d_b[offset+i];

		for (int j=nDim_matrix-1; j>i; j--)
		{
			d_x[offset+i] -= d_A[offset+i+j*nDim_matrix]*d_x[offset+j];
		}
        d_x[offset+i] = d_x[offset+i]/d_A[offset+i+i*nDim_matrix];
	}
/*	do i=ndim,1,-1
		x(i)=b(i)
		do j=ndim,i+1,-1
			x(i)=x(i)-A(i,j)*x(j)
		end do
		x(i)=x(i)/A(i,i)
	end do
*/
}
