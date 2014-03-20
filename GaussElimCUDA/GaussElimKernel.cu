// Gaussian Elimination Function

__global__ void ForwardElimKernel(double d_A[3][4], double d_piv[3], nDim);

void GaussElim(double h_A[3][4], double h_b[3], double h_x[3], const int nDim)
{
	double d_A[3][4], d_piv[3];

	// Allocate memory on device
	//cudaMalloc(d_A,sizeof(float)*(numvar)*(numvar+1));
	//cudaMalloc(d_piv,sizeof(float)*(numvar)*(numvar+1));

	// Copy data from host to device
	//cudaMemcpy(a_d, temp_h, sizeof(float)*numvar*(numvar+1),cudaMemcpyHostTo Device);

	// Define thread block size
	//dim3 dimBlock(numvar+1,numvar,1);
	//dim3 dimGrid(1,1,1);

	// Forward elimination kernel
	ForwardElimKernel<<<dimGrid , dimBlock>>>(d_A, d_piv, nDim);

	// Copy data from device to host
	//cudaMemcpy(temp1_h,b_d,sizeof(float)*numvar*(numvar+1),cudaMemcpyDeviceT oHost);

	// Free memory on device
	cudaFree(d_A);
	cudaFree(d_piv);

	// Backward substitution
	for (int i = 0; nDim-1; i++)
		h_b[i] = h_A[i][nDim];	
	for (int i = nDim-1; i >= 0; i--)
	{
		h_x[i] = h_b[i];
		for (int j = nDim-1; j >= i+1; j--)
			h_x[i] = h_x[i] - h_A[i][j]*h_x[j];
		h_x[i] = h_x[i]/h_A[i][i];
	}
}

