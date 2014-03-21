
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
    int i = blockDim.y*blockIdx.y + threadIdx.y;
    int j = blockDim.x*blockIdx.x + threadIdx.x;

    // Parallel forward elimination
    for (int k = 0; k < Nsize-1; k++)
    {
        d_Piv[i] = d_Aug[i][k]/d_Aug[k][k];
        __syncthreads();
        if ((i>k) && (i<Nsize) && (j>=k) && (j<=Nsize))
            d_Aug[i][j] -= d_Piv[i]*d_Aug[k][j];
        __syncthreads();
    }
}
