// Gaussian Elimination

#include <iostream>
#include "GaussElimCPU.h"

int main()
{
	using namespace std;

	// Hard code inputs (temporary)
	const int nDim = 3;
	double A[3][3] = {
	{6, -1, -2},
	{-6, 13, -6},
	{-2, -1, 6}
	};
	double b[3] = {3, 1, 3};
	double x[3] = {0, 0, 0};

	// Perform Gaussian elimination
	GaussElim(A, b, x, nDim);

	// Output solution to console
	cout << "Solution:" << endl;
	for (int i = 0; i <= nDim-1; i++)
		cout << "x(" << i << ") = " << x[i] << endl;

	return 0;
}
