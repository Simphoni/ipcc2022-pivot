#ifndef OPT2D_H
#define OPT2D_H

#define THREADS_2D 64

#include <math.h>
void solve2d(const int n, const int dim, const int M, double *coord, // ND layout
             double *maxVal, int *maxComb, double *minVal, int *minComb);
#endif
