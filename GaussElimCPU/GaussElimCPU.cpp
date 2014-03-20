// Gaussian Elimination Function

void GaussElim(double A[3][3], double b[3], double x[3], const int nDim)
{
	// Forward elimination
	for (int k = 0; k <= nDim-2; k++)
	{
		for (int i = k+1; i <= nDim-1; i++)
		{
			double pivot = A[i][k]/A[k][k];
			for (int j = k; j <= nDim-1; j++)
				A[i][j] = A[i][j] - pivot*A[k][j];
			b[i] = b[i] - pivot*b[k];
		}
	}

	// Backward substitution
	for (int i = nDim-1; i >= 0; i--)
	{
		x[i] = b[i];
		for (int j = nDim-1; j >= i+1; j--)
			x[i] = x[i] - A[i][j]*x[j];
		x[i] = x[i]/A[i][i];
	}
}
