#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <algorithm>
#include <math.h>
#include "solver.h"
#define uint unsigned long long

using namespace std;

#define THREADS_2D 42

int cmp(const void* a, const void* b) {
	return (*(double*)a) - (*(double*)b) < 0 ? -1 : 1;
}

#define calc(x,y) pow((x) - (y), 2)

// k = 2
void pivot_solver_2d(int n, int dim, double* coord, // ND layout
		double* maxVal, int* maxComb, double* minVal, int* minComb, int nprocs, int rank) {
	int nthreads = nprocs * THREADS_2D;
	int average = (n + 1) / nthreads;
	average = (average + 1) / 2;

	// malloc pools for collection
	double* maxValAll = (double*)malloc(sizeof(double) * BUFF * THREADS_2D);
	int* maxCombAll = (int*)malloc(sizeof(int) * BUFF * THREADS_2D * 2);
	double* minValAll = (double*)malloc(sizeof(double) * BUFF * THREADS_2D);
	int* minCombAll = (int*)malloc(sizeof(int) * BUFF * THREADS_2D * 2);

#pragma omp parallel num_threads(THREADS_2D)
	{
		int id = omp_get_thread_num() + THREADS_2D * rank;
		int thrid = omp_get_thread_num();
		int jobid[10], cnt = 0;
		char* vis = (char*)malloc(sizeof(char) * n);
		memset(vis, 0, sizeof(char) * n);
		bool running = (average * id) <= (n - average * id - 1);
		for (int xid = 0; xid < id; xid ++)
			for (int i = 0; i < average; i++) {
				vis[average * xid + i] = 1;
				vis[n - average * xid - i - 1] = 1;
			}
		for (int i = 0; i < average; i++) {
			if (!vis[average * id + i]) {
				jobid[cnt++] = average * id + i;
				vis[average * id + i] = 1;
			}
			if (!vis[n - average * id - i - 1]) {
				jobid[cnt++] = n - average * id - i - 1;
				vis[n - average * id - i - 1] = 1;
			}
		}
		free(vis);

		double* maxVal = (double*)malloc(sizeof(double) * BUFF);
		int* maxComb = (int*)malloc(sizeof(int) * BUFF * 2);
		double* minVal = (double*)malloc(sizeof(double) * BUFF);
		int* minComb = (int*)malloc(sizeof(int) * BUFF * 2);
		double* c = (double*)malloc(sizeof(double) * n);
		double* a = (double*)malloc(sizeof(double) * n);
		double* b = (double*)malloc(sizeof(double) * n);
		memset(maxVal, 0, sizeof(double) * BUFF);
		memset(minVal, 0x7f, sizeof(double) * BUFF);

		for (int iter = 0; iter < cnt; iter++) {
			if (!running) continue;
			int u = jobid[iter];
			//printf("[%d] %d\n", id, u);
			// calc pivot1
			for (int i = 0; i < n; i++) {
				double sum = 0;
				for (int j = 0; j < dim; j++)
					sum += calc(coord[u * dim + j], coord[i * dim + j]);
				c[i] = sqrt(sum);
			}

			for (int v = u + 1; v < n; v++) {
				// calc pivot2
				for (int i = 0; i < n; i++) {
					double sum = 0;
					for (int j = 0; j < dim; j++)
						sum += calc(coord[v * dim + j], coord[i * dim + j]);
					sum = sqrt(sum);
					a[i] = sum - c[i];
					b[i] = sum + c[i];
				}
				// qsort(a, n, sizeof(double), cmp);
				// qsort(b, n, sizeof(double), cmp);
				sort(a, a + n); sort(b, b + n);

				double ans = 0, asum = 0, bsum = 0;
				for (int i = 0; i < n; i++) {
					ans += (a[i] + b[i]) * i - asum - bsum;
					asum += a[i]; bsum += b[i];
				}
				if (ans > maxVal[0]) {
					int j = 1;
					for (j = 1; j < BUFF; j++) {
						if (ans < maxVal[j]) break;
						else {
							maxVal[j - 1] = maxVal[j];
							maxComb[2 * (j - 1)] = maxComb[2 * j];
							maxComb[2 * (j - 1) + 1] = maxComb[2 * j + 1];
						}
					}
					j--;
					maxVal[j] = ans;
					maxComb[2 * j] = u;
					maxComb[2 * j + 1] = v;
				}
				if (ans < minVal[0]) {
					int j = 1;
					for (j = 1; j < BUFF; j++) {
						if (ans > minVal[j]) break;
						else {
							minVal[j - 1] = minVal[j];
							minComb[2 * (j - 1)] = minComb[2 * j];
							minComb[2 * (j - 1) + 1] = minComb[2 * j + 1];
						}
					}
					j--;
					minVal[j] = ans;
					minComb[2 * j] = u;
					minComb[2 * j + 1] = v;
				}
			}
		}
		memcpy(maxValAll + BUFF * thrid, maxVal, sizeof(double) * BUFF);
		memcpy(maxCombAll + BUFF * thrid * 2, maxComb, sizeof(int) * BUFF * 2);
		memcpy(minValAll + BUFF * thrid, minVal, sizeof(double) * BUFF);
		memcpy(minCombAll + BUFF * thrid * 2, minComb, sizeof(int) * BUFF * 2);
		free(maxVal);
		free(maxComb);
		free(minVal);
		free(minComb);
		free(a);
		free(b);
		free(c);
	}
	// threads forced sync
	static int ptr[THREADS_2D + 1];

	for (int i = 0; i < THREADS_2D; i++) ptr[i] = i * BUFF + BUFF - 1;
	for (int i = 0; i < BUFF; i++) {
		int id = 0;
		for (int j = 0; j < THREADS_2D; j++)
			if (maxValAll[ptr[j]] > maxValAll[ptr[id]]) id = j;
		int p = ptr[id];
		ptr[id] --;
		maxVal[i] = maxValAll[p];
		maxComb[i * 2] = maxCombAll[p * 2];
		maxComb[i * 2 + 1] = maxCombAll[p * 2 + 1];
	}

	for (int i = 0; i < THREADS_2D; i++) ptr[i] = i * BUFF + BUFF - 1;
	for (int i = 0; i < BUFF; i++) {
		int id = 0;
		for (int j = 0; j < THREADS_2D; j++)
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
}
