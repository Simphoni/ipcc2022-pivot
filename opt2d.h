#ifndef OPT2D_H
#define OPT2D_H

#define BUFF 1000
#define THREADS_2D 32

void solve2d(const int n, const int dim, double *coord, // ND layout
             double *maxVal, int *maxComb, double *minVal, int *minComb);
#endif
