#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "opt2d.h"
#define uint unsigned long long

int cmp(const void *a, const void *b) {
  return (*(double*)a) < (*(double*)b);
}

#define calc(x,y) pow((x) - (y), 2)

// k = 2
void solve2d(const int n, const int dim, double *coord, // ND layout
             double *maxVal, int *maxComb, double *minVal, int *minComb) {
  int totCombs = n * (n - 1) / 2;
  int average = totCombs / THREADS_2D;
  static int assign[THREADS_2D + 1];
  // job assign
  { int tmp = 1, sum = 0;
    for (int i = 1; i <= n; i ++) {
      sum += n - i;
      if (sum >= average * 0.99) {
        assign[tmp] = i;
        tmp ++;
        /* fprintf(stderr, "%.3lf\n", 1. * sum / average); */
        sum = 0;
      }
      if (tmp == THREADS_2D) break;
    }
    assign[0] = 0;
    assign[THREADS_2D] = n;
  }
  // malloc pools for collection
  double *maxValAll = (double*)malloc(sizeof(double) * BUFF * THREADS_2D);
  int    *maxCombAll = (int*)malloc(sizeof(int) * BUFF * THREADS_2D * 2);
  double *minValAll = (double*)malloc(sizeof(double) * BUFF * THREADS_2D);
  int    *minCombAll = (int*)malloc(sizeof(int) * BUFF * THREADS_2D * 2);
  double *cAll = (double*)malloc(sizeof(double) * n * THREADS_2D);
  double *aAll = (double*)malloc(sizeof(double) * n * THREADS_2D);
  double *bAll = (double*)malloc(sizeof(double) * n * THREADS_2D);

#pragma omp parallel num_threads(THREADS_2D)
#pragma omp proc_bind(close)
  {
    const int id = omp_get_thread_num();
    int lbound = assign[id];
    int rbound = assign[id + 1];

    double *maxVal = maxValAll + BUFF * id;
    int    *maxComb = maxCombAll + BUFF * id * 2;
    double *minVal = minValAll + BUFF * id;
    int    *minComb = minCombAll + BUFF * id * 2;
    double *c = cAll + n * id;
    double *a = aAll + n * id;
    double *b = bAll + n * id;
    
    for (int u = lbound; u < rbound; u ++) {
      printf("[%d]: %d\n", id, u);
      for (int i = 0; i < n; i ++) {
        double sum = 0;
        for (int j = 0; j < dim; j ++)
          sum += calc(coord[u * 2 + j], coord[i * 2 + j]);
        c[i] = sqrt(sum);
      
        for (int v = u; v < n; v ++) {
          
          for (int i = 0; i < n; i ++) {
            double sum = 0;
            for (int j = 0; j < dim; j ++)
              sum += calc(coord[v * 2 + j], coord[i * 2 + j]);
            sum = sqrt(sum);
            a[i] = fabs(sum - c[i]) * .5;
            b[i] = fabs(sum + c[i]) * .5;
          }
          qsort(a, n, sizeof(double), cmp);
          qsort(b, n, sizeof(double), cmp);
          double ans = 0, asum = 0, bsum = 0;
          for (int i = 0; i < n; i ++) {
            ans += a[i] * i - asum;
            ans += b[i] * i - bsum;
            asum += a[i]; bsum += b[i];
          }
          if (ans > maxVal[0]) {
            int j = 1;
            for (int j = 1; j < BUFF; j ++) {
              if (ans < maxVal[j]) break;
              else {
                maxVal[j - 1] = maxVal[j];
                maxComb[2 * (j - 1)] = maxComb[2 * j];
                maxComb[2 * (j - 1) + 1] = maxComb[2 * j + 1];
              }
            }
            j --;
            maxVal[j] = ans;
            maxComb[2 * j] = u;
            maxComb[2 * j + 1] = v;
          }
          if (ans < minVal[0]) {
            int j = 1;
            for (int j = 1; j < BUFF; j ++) {
              if (ans < minVal[j]) break;
              else {
                minVal[j - 1] = minVal[j];
                minComb[2 * (j - 1)] = minComb[2 * j];
                minComb[2 * (j - 1) + 1] = minComb[2 * j + 1];
              }
            }
            j --;
            minVal[j] = ans;
            minComb[2 * j] = u;
            minComb[2 * j + 1] = v;
          }
        }
      }
    }
  }
  static int ptr[THREADS_2D + 1];
  
  for (int i = 0; i < THREADS_2D; i ++) ptr[i] = i * BUFF + BUFF - 1;
  for (int i = 0; i < BUFF; i ++) {
    int id = 0;
    for (int j = 0; j < THREADS_2D; j ++)
      if (maxValAll[ptr[j]] > maxValAll[ptr[id]]) id = j;
    int p = ptr[id];
    ptr[id] --;
    maxVal[i] = maxValAll[p];
    maxComb[i * 2] = maxCombAll[p * 2];
    maxComb[i * 2 + 1] = maxCombAll[p * 2 + 1];
  }

  for (int i = 0; i < THREADS_2D; i ++) ptr[i] = i * BUFF + BUFF - 1;
  for (int i = 0; i < BUFF; i ++) {
    int id = 0;
    for (int j = 0; j < THREADS_2D; j ++)
      if (minValAll[ptr[j]] < minValAll[ptr[id]]) id = j;
    int p = ptr[id];
    ptr[id] --;
    minVal[i] = minValAll[p];
    minComb[i * 2] = minCombAll[p * 2];
    minComb[i * 2 + 1] = minCombAll[p * 2 + 1];
  }
  
  free(maxValAll);
  free(maxCombAll);
  free(minValAll);
  free(minCombAll);
  free(cAll);
  free(aAll);
  free(bAll);
}
