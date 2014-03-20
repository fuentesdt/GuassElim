
/*
 * Device code
 */
__global__ 
void GaussSolve(
         int const Nsize,
         const double* d_Matrix,
         const double* d_RHS,
               double* d_Soln)
{
    /* Calculate the global linear index, assuming a 1-d grid. */
    int const idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < Nsize)
      printf("idx=%d A[%d][%d]=%f b[%d]=%f\n",idx,idx,idx,d_Matrix[idx+Nsize*idx], idx,d_RHS[idx]);
}
