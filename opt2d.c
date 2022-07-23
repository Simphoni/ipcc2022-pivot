#include <stdio.h>
#include <string.h>
#include "opt2d.h"
#define uint unsigned long long

static void sort(uint *a, uint *swp, int n) {
  static int buc0[1 << 8], buc1[1 << 8], buc2[1 << 8], buc3[1 << 8];
  memset(buc0, 0, sizeof buc0);
  memset(buc1, 0, sizeof buc0);
  memset(buc2, 0, sizeof buc0);
  memset(buc3, 0, sizeof buc0);
  for (int i = 0; i < n; ++ i) {
    ++ buc0[a[i]       & 255];
    ++ buc1[a[i] >>  8 & 255];
    ++ buc2[a[i] >> 16 & 255];
    ++ buc3[a[i] >> 24 & 255];
  }
  for (int i = 1; i < 256; ++ i) {
    buc0[i] += buc0[i - 1];
    buc1[i] += buc1[i - 1];
    buc2[i] += buc2[i - 1];
    buc3[i] += buc3[i - 1];
  }
  // round 1
  uint *ptr = a + n - 1;
  for (int iter = n >> 3; iter; iter --) {
    swp[-- buc0[ptr[ 0] & 255]] = ptr[ 0];
    swp[-- buc0[ptr[-1] & 255]] = ptr[-1];
    swp[-- buc0[ptr[-2] & 255]] = ptr[-2];
    swp[-- buc0[ptr[-3] & 255]] = ptr[-3];
    swp[-- buc0[ptr[-4] & 255]] = ptr[-4];
    swp[-- buc0[ptr[-5] & 255]] = ptr[-5];
    swp[-- buc0[ptr[-6] & 255]] = ptr[-6];
    swp[-- buc0[ptr[-7] & 255]] = ptr[-7];
    ptr -= 8;
  }
  while (ptr >= a) {
    swp[-- buc0[ptr[0] & 255]] = ptr[0];
    ptr --;
  }
  // round 2
  ptr = swp + n - 1;
  for (int iter = n >> 3; iter; iter --) {
    a[-- buc1[ptr[ 0] >> 8 & 255]] = ptr[ 0];
    a[-- buc1[ptr[-1] >> 8 & 255]] = ptr[-1];
    a[-- buc1[ptr[-2] >> 8 & 255]] = ptr[-2];
    a[-- buc1[ptr[-3] >> 8 & 255]] = ptr[-3];
    a[-- buc1[ptr[-4] >> 8 & 255]] = ptr[-4];
    a[-- buc1[ptr[-5] >> 8 & 255]] = ptr[-5];
    a[-- buc1[ptr[-6] >> 8 & 255]] = ptr[-6];
    a[-- buc1[ptr[-7] >> 8 & 255]] = ptr[-7];
    ptr -= 8;
  }
  while (ptr >= swp) {
    a[-- buc1[ptr[0] >> 8 & 255]] = ptr[0];
    ptr --;
  }
  // round 3
  ptr = a + n - 1;
  for (int iter = n >> 3; iter; iter --) {
    swp[-- buc2[ptr[ 0] >> 16 & 255]] = ptr[ 0];
    swp[-- buc2[ptr[-1] >> 16 & 255]] = ptr[-1];
    swp[-- buc2[ptr[-2] >> 16 & 255]] = ptr[-2];
    swp[-- buc2[ptr[-3] >> 16 & 255]] = ptr[-3];
    swp[-- buc2[ptr[-4] >> 16 & 255]] = ptr[-4];
    swp[-- buc2[ptr[-5] >> 16 & 255]] = ptr[-5];
    swp[-- buc2[ptr[-6] >> 16 & 255]] = ptr[-6];
    swp[-- buc2[ptr[-7] >> 16 & 255]] = ptr[-7];
    ptr -= 8;
  }
  while (ptr >= a) {
    swp[-- buc2[ptr[0] >> 16 & 255]] = ptr[0];
    ptr --;
  }
  // round 4
  ptr = swp + n - 1;
  for (int iter = n >> 3; iter; iter --) {
    a[-- buc3[ptr[ 0] >> 24 & 255]] = ptr[ 0];
    a[-- buc3[ptr[-1] >> 24 & 255]] = ptr[-1];
    a[-- buc3[ptr[-2] >> 24 & 255]] = ptr[-2];
    a[-- buc3[ptr[-3] >> 24 & 255]] = ptr[-3];
    a[-- buc3[ptr[-4] >> 24 & 255]] = ptr[-4];
    a[-- buc3[ptr[-5] >> 24 & 255]] = ptr[-5];
    a[-- buc3[ptr[-6] >> 24 & 255]] = ptr[-6];
    a[-- buc3[ptr[-7] >> 24 & 255]] = ptr[-7];
    ptr -= 8;
  }
  while (ptr >= swp) {
    a[-- buc3[ptr[0] >> 24 & 255]] = ptr[0];
    ptr --;
  }
}

// k = 2
void solve2d(const int n, const int dim, const int M, double *coord, // ND layout
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
        fprintf(stderr, "%.3lf\n", 1. * sum / average);
        sum = 0;
      }
      if (tmp == THREADS_2D) break;
    }
    assign[0] = 0;
    assign[THREADS_2D] = n;
  }

#pragma omp parallel num_threads(THREADS_2D) proc_bind(close)
  {
    
  }
}
