
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
