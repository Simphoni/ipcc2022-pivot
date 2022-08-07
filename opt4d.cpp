#include <bits/stdc++.h>
#include <immintrin.h>
#include "mpi.h"
#include "omp.h"
#include "solver.h"

#define OMP_THREADS 64

using namespace std;

inline int64_t calc_comb(int n, int k) {
    int64_t ret = 1;
    for (int i = 1; i <= k; i++) ret *= n + 1 - i;
    for (int i = 1; i <= k; i++) ret /= i;
    return ret;
}

//__attribute__((always_inline))
inline double calc(double x) { return x * x; }


#define NPIVOTS 4

void pivot_solver_4d(int n, int dim, double* coord, // ND layout
    double* maxVal, int* maxComb, double* minVal, int* minComb, int nprocs, int rank) {
    // job assign
    int bar[128][2], cnt = 0;
    memset(bar, 0, sizeof(bar));
    int nthreads = nprocs * OMP_THREADS;
    int64_t average = calc_comb(n, NPIVOTS) / nthreads + 1;
    printf("%llu\n", average);
    fflush(stdout);
    int64_t cur = 0;
    for (int i = 1; i <= n; i++)
        for (int j = i + 1; j <= n; j++) {
            cur += calc_comb(n - j, NPIVOTS - 2);
            if (cur >= average) {
                cur = 0;
                ++cnt;
                bar[cnt][0] = i - 1;
                bar[cnt][1] = j - 1;
            }
        }

    return;

    // global var
    double* maxValAll = (double*)malloc(sizeof(double) * BUFF * OMP_THREADS);
    int* maxCombAll = (int*)malloc(sizeof(int) * BUFF * OMP_THREADS * NPIVOTS);
    double* minValAll = (double*)malloc(sizeof(double) * BUFF * OMP_THREADS);
    int* minCombAll = (int*)malloc(sizeof(int) * BUFF * OMP_THREADS * NPIVOTS);

#pragma omp parallel num_threads(OMP_THREADS)
    {
        int id = rank * OMP_THREADS + omp_get_thread_num();
        int thrid = omp_get_thread_num();
        // local var
        double* maxVal = (double*)malloc(sizeof(double) * BUFF);
        int* maxComb = (int*)malloc(sizeof(int) * BUFF * NPIVOTS);
        double* minVal = (double*)malloc(sizeof(double) * BUFF);
        int* minComb = (int*)malloc(sizeof(int) * BUFF * NPIVOTS);
        double* rebuilt = (double*)aligned_alloc(32, sizeof(double) * NPIVOTS * n);
        double tmp[4];

        int done = 0;

        for (int x = bar[id][0]; x <= bar[id + 1][0]; x++) {
            int yl = x + 1, yr = n;
            if (x == bar[id][0]) yl = max(yl, bar[id][1] + 1);
            if (x == bar[id + 1][0]) yr = min(yr, bar[id + 1][1]);

            for (int y = yl; y <= yr; y++)
                for (int z = y + 1; z <= n; z++)
                    for (int a = y + 1; a <= n; a++) {
                        double ans = 0;
                        for (int i = 0; i < n; i++) {
                            double sumx = 0, sumy = 0, sumz = 0, suma = 0;// sumb = 0;
                            for (int j = 0; j < dim; j++) {
                                double p = coord[i * dim + j];
                                sumx += calc(coord[x * dim + j] - p);
                                sumy += calc(coord[y * dim + j] - p);
                                sumz += calc(coord[z * dim + j] - p);
                                suma += calc(coord[a * dim + j] - p);
                                //suma += calc(coord[b * dim + j] - p);
                            }
                            rebuilt[i * NPIVOTS] = sqrt(sumx);
                            rebuilt[i * NPIVOTS + 1] = sqrt(sumy);
                            rebuilt[i * NPIVOTS + 2] = sqrt(sumz);
                            rebuilt[i * NPIVOTS + 3] = sqrt(suma);
                            //rebuilt[i * NPIVOTS + 4] = sqrt(sumb);
                        }
                        for (int i = 0; i < n; i++)
                            for (int j = i + 1; j < n; j++) {
                                __m256d a = _mm256_load_pd(&rebuilt[i * 4]);
                                __m256d b = _mm256_load_pd(&rebuilt[j * 4]);
                                __m256d c = _mm256_sub_pd(a, b);
                                __m256d d = _mm256_sub_pd(b, a);
                                c = _mm256_max_pd(c, d); // abs
                                d = _mm256_permute_pd(c, 0x5); // (b) 0101 (big endian)
                                c = _mm256_max_pd(c, d);
                                d = _mm256_permute4x64_pd(c, 0xb0); // (q) 2300 (big endian)
                                c = _mm256_max_pd(c, d);
                                _mm256_store_pd(tmp, c);
                                ans += tmp[0];
                            }

                        if (ans > maxVal[0]) {
                            int j = 1;
                            for (j = 1; j < BUFF; j++) {
                                if (ans < maxVal[j]) break;
                                else {
                                    maxVal[j - 1] = maxVal[j];
                                    maxComb[4 * (j - 1)] = maxComb[4 * j];
                                    maxComb[4 * (j - 1) + 1] = maxComb[4 * j + 1];
                                    maxComb[4 * (j - 1) + 2] = maxComb[4 * j + 2];
                                    maxComb[4 * (j - 1) + 3] = maxComb[4 * j + 3];
                                }
                            }
                            j--;
                            maxVal[j] = ans;
                            maxComb[4 * j] = x;
                            maxComb[4 * j + 1] = y;
                            maxComb[4 * j + 2] = z;
                            maxComb[4 * j + 3] = a;
                        }

                        if (ans < minVal[0]) {
                            int j = 1;
                            for (j = 1; j < BUFF; j++) {
                                if (ans > minVal[j]) break;
                                else {
                                    minVal[j - 1] = minVal[j];
                                    minComb[4 * (j - 1)] = minComb[4 * j];
                                    minComb[4 * (j - 1) + 1] = minComb[4 * j + 1];
                                    minComb[4 * (j - 1) + 2] = minComb[4 * j + 2];
                                    minComb[4 * (j - 1) + 3] = minComb[4 * j + 3];
                                }
                            }
                            j--;
                            minVal[j] = ans;
                            minComb[4 * j] = x;
                            minComb[4 * j + 1] = y;
                            minComb[4 * j + 2] = z;
                            minComb[4 * j + 3] = a;
                        }
                        done++;
                        if (done % 1000 == 0) {
                            //printf("[%d] %d/%llu\n", id, done, average);
                            //fflush(stdout);
                        }
                    }
        }
        memcpy(maxValAll + BUFF * thrid, maxVal, sizeof(double) * BUFF);
        memcpy(maxCombAll + BUFF * thrid * NPIVOTS, maxComb, sizeof(int) * BUFF * NPIVOTS);
        memcpy(minValAll + BUFF * thrid, minVal, sizeof(double) * BUFF);
        memcpy(minCombAll + BUFF * thrid * NPIVOTS, minComb, sizeof(int) * BUFF * NPIVOTS);
        free(maxVal);
        free(maxComb);
        free(minVal);
        free(minComb);
        free(rebuilt);
    }

    // threads forced sync
    static int ptr[OMP_THREADS + 1];

    for (int i = 0; i < OMP_THREADS; i++) ptr[i] = i * BUFF + BUFF - 1;
    for (int i = 0; i < BUFF; i++) {
        int id = 0;
        for (int j = 0; j < OMP_THREADS; j++)
            if (maxValAll[ptr[j]] > maxValAll[ptr[id]]) id = j;
        int p = ptr[id];
        ptr[id] --;
        maxVal[i] = maxValAll[p];
        maxComb[i * 4] = maxCombAll[p * 4];
        maxComb[i * 4 + 1] = maxCombAll[p * 4 + 1];
        maxComb[i * 4 + 2] = maxCombAll[p * 4 + 2];
        maxComb[i * 4 + 3] = maxCombAll[p * 4 + 3];
    }

    for (int i = 0; i < OMP_THREADS; i++) ptr[i] = i * BUFF + BUFF - 1;
    for (int i = 0; i < BUFF; i++) {
        int id = 0;
        for (int j = 0; j < OMP_THREADS; j++)
            if (minValAll[ptr[j]] < minValAll[ptr[id]]) id = j;
        int p = ptr[id];
        ptr[id] --;
        minVal[i] = minValAll[p];
        minComb[i * 4] = minCombAll[p * 4];
        minComb[i * 4 + 1] = minCombAll[p * 4 + 1];
        minComb[i * 4 + 2] = minCombAll[p * 4 + 2];
        minComb[i * 4 + 3] = minCombAll[p * 4 + 3];
    }

    free(maxValAll);
    free(maxCombAll);
    free(minValAll);
    free(minCombAll);
}