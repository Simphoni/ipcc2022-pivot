#include <bits/stdc++.h>
#include <immintrin.h>
#include <iostream>
#include "mpi.h"
#include "omp.h"
#include "solver.h"

#define MAXN 200
#define MAXK 20

#define int64_t unsigned long long
// #define BUFF 1000

#define INDEX(x, y, n) ((x) * (n) + (y))
#define OMP_THREADS 64

using namespace std;

inline int64_t calc_comb(int n, int k) {
  int64_t ret = 1;
  for (int i = 1; i <= k; i++) ret *= n + 1 - i;
  for (int i = 1; i <= k; i++) ret /= i;
  return ret;
}

inline double calc(double x) { return x * x; }


double *table[MAXN][MAXN];
uint64_t C[2000][MAXK];

void initCombs();
void initTable(int n, double *dist);
void initDist(int n, int dim, double* coord, double *dist);
void getKthPivot(int n, int k, uint64_t cid, int *pivots);

// dis[n][n]: stores original pointwise distance
void pivot_solver_4d(int n, int dim, double* coord, // ND layout
                     double* maxVal, int* maxComb, double* minVal, int* minComb, int nprocs, int rank) {
  double *dist = (double*)malloc(n * n * sizeof(double));
  initDist(n, dim, coord, dist);
  initCombs();
  initTable(n, dist);
}

// build C[][], ignore overflow
void initCombs() {
  memset(C, 0, sizeof C);
  for (int i = 0; i < MAXN; i ++) {
    C[i][0] = 1;
    for (int j = 0; j <= i; j ++)
      C[i][j] = C[i - 1][j - 1] + C[i - 1][j];
  }
}

// build a dist table for all combination of pivots of size 2
void initTable(int n, double *dist) {
  uint64_t njobs = C[n][2];
#pragma omp num_threads (OMP_THREADS)
  {
    int rec[2];
    const int id = omp_get_thread_num();
    for (uint64_t jobid = id; jobid < njobs; jobid += OMP_THREADS) {
      getKthPivot(n, 2, jobid, rec);
      int u = rec[0], v = rec[1];
      
      cerr << jobid << " " << u << " " << v << endl;
      
      double *vec = (double*)malloc(n * n * sizeof(double));
      table[u][v] = vec;
      for (int i = 0; i < n; i ++)
        for (int j = i + 1; j < n; j ++) {
          vec[INDEX(i, j, n)] = vec[INDEX(j, i, n)]
            = min( fabs( dist[INDEX(u, i, n)] - dist[INDEX(u, j, n)] ),
                   fabs( dist[INDEX(v, i, n)] - dist[INDEX(v, j, n)] ) );
        }
    }
  }
}

void initDist(int n, int dim, double* coord, double *dist) {
  for (int i = 0; i < n; i ++)
    for (int j = i; j < n; j ++) {
      double tmp = 0;
      for (int k = 0; k < dim; k ++)
        tmp += calc( coord[INDEX(i, k, dim)] - coord[INDEX(j, k, dim)] );
      dist[INDEX(i, j, n)] = dist[INDEX(j, i, n)] = sqrt(tmp);
    }
}

void getKthPivot(int n, int k, uint64_t cid, int *pivots) {
  // the cid-th combination for `n choose k`
  int last_pivot = -1;
  for (int i = k - 1; i >= 0; i--) {
    int j = last_pivot + 1;
    cid -= C[n - 1 - j][i];
    while (cid >= 0) { // check the case where cid == 0
      ++j;
      cid -= C[n - 1 - j][i];
    }
    cid += C[n - 1 - j][i];
    pivots[k - 1 - i] = j;
    last_pivot = j;
  }
}
