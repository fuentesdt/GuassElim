// Gaussian Elimination

#include <iostream>
#include "GaussElimCPU.h"

int main()
{
	using namespace std;

	// Hard code inputs (temporary)
	const int nDim = 3;
	double h_A[3][4] = {
	{6, -1, -2, 3},
	{-6, 13, -6, 1},
	{-2, -1, 6, 3}
	};
	double h_b[3] = {3, 1, 3};
	double h_x[3] = {0, 0, 0};

	// Perform Gaussian elimination
	GaussElim(h_A, h_b, h_x, nDim);

	// Output solution to console
	cout << "Solution:" << endl;
	for (int i = 0; i <= nDim-1; i++)
		cout << "x(" << i << ") = " << h_x[i] << endl;

	return 0;
}
