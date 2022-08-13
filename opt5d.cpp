#include <bits/stdc++.h>
#include <immintrin.h>
#include <iostream>
#include <x86intrin.h>
#include "mpi.h"
#include "omp.h"
#include "solver.h"

#define OMP_THREADS 64
#define MAXN 110
#define MAXK 10
#define K 5

#define int64_t long long
// #define BUFF 1000

#define INDEX(x, y, n) ((x) * (n) + (y))

using namespace std;

inline double calc(double x) { return x * x; }

double dist[MAXN][MAXN];
double* table[MAXN][MAXN];
int64_t C[2000][MAXK];

void initCombs();
void initTable(int n);
void initDist(int n, int dim, double* coord);
void getKthPivot(int n, int k, int64_t cid, int* pivots);

// dis[n][n]: stores original pointwise distance
void pivot_solver_common(int n, int dim, double* coord, // ND layout
  double* maxVal, int* maxComb, double* minVal, int* minComb, int nprocs, int rank) {
  initDist(n, dim, coord);
  initCombs();
  initTable(n);

  // malloc pools for collection
  double* maxValAll = (double*)malloc(sizeof(double) * BUFF * OMP_THREADS);
  int* maxCombAll = (int*)malloc(sizeof(int) * BUFF * OMP_THREADS * K);
  double* minValAll = (double*)malloc(sizeof(double) * BUFF * OMP_THREADS);
  int* minCombAll = (int*)malloc(sizeof(int) * BUFF * OMP_THREADS * K);

#pragma omp parallel num_threads(OMP_THREADS)
  {
    const int nthr = OMP_THREADS * nprocs;
    const int id = omp_get_thread_num() + OMP_THREADS * rank;
    const int thrid = omp_get_thread_num();

    double* maxVal = maxValAll + BUFF * thrid;
    int* maxComb = maxCombAll + BUFF * thrid * K;
    double* minVal = minValAll + BUFF * thrid;
    int* minComb = minCombAll + BUFF * thrid * K;
    memset(maxVal, 0, sizeof(double) * BUFF);
    memset(minVal, 0x7f, sizeof(double) * BUFF);

    double* acc = (double*)aligned_alloc(32, sizeof(double) * n * (n - 1) / 2);

    const int k_enumed = K - 1;
    int comb[20];

    int64_t njobs = C[n - 1][k_enumed];
    const int veclen = n * (n - 1) / 2;
    // collect
    int64_t load = 0, _load = 0;

    for (int64_t iter = 0; iter < njobs; iter += nthr) {
      int64_t jobid = iter + (id + (iter / nthr)) % nthr;
      if (jobid >= njobs) break;
      getKthPivot(n - 1, k_enumed, jobid, comb);

      double* src = table[comb[0]][comb[1]]; // comb[0] < comb[1]
      double* crd = dist[comb[2]], * _crd = dist[comb[3]];
      _load++;
      int cnt = 0;
      for (int i = 0; i < n; i++)
#pragma unroll
        for (int j = i + 1; j < n; j++, cnt++)
          acc[cnt] = max(src[cnt], max(fabs(crd[i] - crd[j]), fabs(_crd[i] - _crd[j])));

      // enum final dim
      for (int x = comb[3] + 1; x < n; x++) {
        load++;
        double ans = 0;
        double* _src = table[x][x];
        {
          int i = 0;
          __m256d sum = _mm256_setzero_pd();
#pragma unroll
          for (i = 0; i + 3 < veclen; i += 4) {
            __m256d opa = _mm256_load_pd(acc + i);
            __m256d opb = _mm256_load_pd(_src + i);
            opa = _mm256_max_pd(opa, opb);
            sum = _mm256_add_pd(opa, sum);
          }
          double str[4];
          _mm256_storeu_pd(str, sum);
          ans = str[0] + str[1] + str[2] + str[3];
          for (; i < veclen; i++) ans += max(acc[i], _src[i]);
          ans *= 2;
        }

        // update
        if (ans > maxVal[0]) {
          int j = 1;
          for (j = 1; j < BUFF; j++) {
            if (ans < maxVal[j]) break;
            else {
              maxVal[j - 1] = maxVal[j];
              for (int u = 0; u < K; u++)
                maxComb[K * (j - 1) + u] = maxComb[K * j + u];
            }
          }
          j--;
          maxVal[j] = ans;
          for (int u = 0; u < k_enumed; u++)
            maxComb[K * j + u] = comb[u];
          maxComb[K * j + K - 1] = x;
        }
        if (ans < minVal[0]) {
          int j = 1;
          for (j = 1; j < BUFF; j++) {
            if (ans > minVal[j]) break;
            else {
              minVal[j - 1] = minVal[j];
              for (int u = 0; u < K; u++)
                minComb[K * (j - 1) + u] = minComb[K * j + u];
            }
          }
          j--;
          minVal[j] = ans;
          for (int u = 0; u < k_enumed; u++)
            minComb[K * j + u] = comb[u];
          minComb[K * j + K - 1] = x;
        }

      }
    }
    /*
    if (id <= 20)
      printf("[%d]: %lld %lld\n", id, load, _load);
       */
  }
  // merge
  static int ptr[OMP_THREADS + 1];

  for (int i = 0; i < OMP_THREADS; i++) ptr[i] = i * BUFF + BUFF - 1;
  for (int i = 0; i < BUFF; i++) {
    int id = 0;
    for (int j = 0; j < OMP_THREADS; j++)
      if (maxValAll[ptr[j]] > maxValAll[ptr[id]]) id = j;
    int p = ptr[id];
    ptr[id] --;
    maxVal[i] = maxValAll[p];
    for (int u = 0; u < K; u++)
      maxComb[i * K + u] = maxCombAll[p * K + u];
  }

  for (int i = 0; i < OMP_THREADS; i++) ptr[i] = i * BUFF + BUFF - 1;
  for (int i = 0; i < BUFF; i++) {
    int id = 0;
    for (int j = 0; j < OMP_THREADS; j++)
      if (minValAll[ptr[j]] < minValAll[ptr[id]]) id = j;
    int p = ptr[id];
    ptr[id] --;
    minVal[i] = minValAll[p];
    for (int u = 0; u < K; u++)
      minComb[i * K + u] = minCombAll[p * K + u];
  }
  free(maxValAll);
  free(maxCombAll);
  free(minValAll);
  free(minCombAll);
}

// build C[][], ignore overflow
void initCombs() {
  memset(C, 0, sizeof C);
  C[0][0] = 1;
  for (int i = 1; i < 1000; i++) {
    C[i][0] = 1;
    for (int j = 1; j < MAXK; j++)
      C[i][j] = C[i - 1][j - 1] + C[i - 1][j];
  }
}

// build a dist table for all combination of pivots of size 2
void initTable(int n) {
  int64_t njobs = C[n + 1][2]; // allow pivot_1 == pivot_2
  int ps = n * (n - 1) / 2;
  ps = ps / 4 * 4 + 4;
  double* pool = (double*)aligned_alloc(32, njobs * ps * sizeof(double));
#pragma omp parallel num_threads(OMP_THREADS)
  {
    int rec[2];
    const int id = omp_get_thread_num();
    for (int64_t jobid = id; jobid < njobs; jobid += OMP_THREADS) {
      getKthPivot(n + 1, 2, jobid, rec);
      int u = rec[0], v = rec[1] - 1;
      // use u,v as pivot

      double* vec = pool + jobid * ps;
      table[u][v] = vec;
      int cnt = 0;
      for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
          vec[cnt++] = max(
            fabs(dist[u][i] - dist[u][j]),
            fabs(dist[v][i] - dist[v][j]));
        }
    }
  }
}

void initDist(int n, int dim, double* coord) {
  for (int i = 0; i < n; i++)
    for (int j = i; j < n; j++) {
      double tmp = 0;
      for (int k = 0; k < dim; k++)
        tmp += calc(coord[INDEX(i, k, dim)] - coord[INDEX(j, k, dim)]);
      dist[i][j] = dist[j][i] = sqrt(tmp);
    }
}

void getKthPivot(int n, int k, int64_t cid, int* pivots) {
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
