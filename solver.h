#ifndef PIVOT_SOLVER_H
#define PIVOT_SOLVER_H

#define BUFF 1000

void pivot_solver_2d(int n, int dim, double* coord, // ND layout
    double* maxVal, int* maxComb, double* minVal, int* minComb, int nprocs, int rank);

void pivot_solver_4d(int n, int dim, double* coord, // ND layout
    double* maxVal, int* maxComb, double* minVal, int* minComb, int nprocs, int rank);

void merge(double* maxValAll, int* maxCombAll, double* minvalAll, int* minCombAll,
    double* maxVal, int* maxComb, double* minVal, int* minComb, int N, int k);

void commun(int k,
    double* maxVal, int* maxComb, double* minVal, int* minComb, int nprocs, int rank);

#endif // PIVOT_SOLVER_H
