
/*
 * Device code
 */

__global__ 
void GaussSolve(
         int const Nsize,
         double* d_Aug,
         double* d_Piv)
{
    // Assign matrix elements to blocks and threads
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    // Parallel forward elimination
    for (int k = 0; k < Nsize-1; k++)
    {
        d_Piv[i] = d_Aug[i%Nsize+k*Nsize]/d_Aug[k*(Nsize+1)];
        __syncthreads();
        if (((i%Nsize)>k) && ((i/Nsize/*+1*/)>=k) && ((i/Nsize/*+1*/)<=Nsize))
            d_Aug[i] -= d_Piv[i]*d_Aug[i-(i%Nsize)+k];
        __syncthreads();
    }
}

__global__ void ParallelGaussElim()
{
	// Assign image pixels to blocks and threads
	int i_image = blockDim.y*blockIdx.y + threadIdx.y;
	int j_image = blockDim.x*blockIdx.x + threadIdx.x;

	int offset = (j_image + i_image*nDim_image)*nDim_mat*nDim_mat;

	// Gauss elimination
	for (int k=0; k<nDim-1; k++)
	{
		for (int i=k+1;	i<nDim; i++)
		{
			pivot = d_A[offset+i+k*nDim_mat]/d_A[offset+k+k*nDim_mat];
			for (int j=k; j<nDim; j++)
			{
				d_A[offset+i+j*nDim_mat] -= pivot*d_A[offset+k+j*nDim_mat];
			}
			d_b[offset+i] -= pivot*d_b[offset+i];
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

	for (int i=nDim-1; i>=0; i--)
	{

		for (int j=nDim-1; j>i+1; j--)
		{

		}

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
