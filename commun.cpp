#include <bits/stdc++.h>
#include "mpi.h"
#include "solver.h"

void merge(double* maxValAll, int* maxCombAll, double* minValAll, int* minCombAll,
    double* maxVal, int* maxComb, double* minVal, int* minComb, int N, int k) {
    // big -> small
    static int ptr[256];
    for (int i = 0; i < N; i++) ptr[i] = i * BUFF;
    for (int i = 0; i < BUFF; i++) {
        int id = 0;
        for (int j = 0; j < N; j++)
            if (maxValAll[ptr[j]] > maxValAll[ptr[id]]) id = j;
        int p = ptr[id];
        ptr[id] ++;
        maxVal[i] = maxValAll[p];
        for (int j = 0; j < k; j++)
            maxComb[i * k + j] = maxCombAll[p * k + j];
    }

    for (int i = 0; i < N; i++) ptr[i] = i * BUFF;
    for (int i = 0; i < BUFF; i++) {
        int id = 0;
        for (int j = 0; j < N; j++)
            if (minValAll[ptr[j]] < minValAll[ptr[id]]) id = j;
        int p = ptr[id];
        ptr[id] ++;
        minVal[i] = minValAll[p];
        for (int j = 0; j < k; j++)
            minComb[i * k + j] = minCombAll[p * k + j];
    }
}

void commun(int k, double* maxVal, int* maxComb, double* minVal, int* minComb, int nprocs, int rank) {
    if (nprocs <= 1) return;
    if (rank == 0) {
        double* maxValAll = (double*)malloc(sizeof(double) * BUFF * 2);
        int* maxCombAll = (int*)malloc(sizeof(int) * BUFF * k * 2);
        double* minValAll = (double*)malloc(sizeof(double) * BUFF * 2);
        int* minCombAll = (int*)malloc(sizeof(int) * BUFF * k * 2);

        memcpy(maxValAll, maxVal, sizeof(double) * BUFF);
        memcpy(maxCombAll, maxComb, sizeof(int) * BUFF * k);
        memcpy(minValAll, minVal, sizeof(double) * BUFF);
        memcpy(minCombAll, minComb, sizeof(int) * BUFF * k);

        MPI_Recv(maxValAll + BUFF, BUFF, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(minValAll + BUFF, BUFF, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(maxCombAll + BUFF * k, BUFF * k, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(minCombAll + BUFF * k, BUFF * k, MPI_INT, 1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        merge(maxValAll, maxCombAll, minValAll, minCombAll,
            maxVal, maxComb, minVal, minComb, 2, k);
        free(maxValAll);
        free(maxCombAll);
        free(minValAll);
        free(minCombAll);
    }
    else {
        MPI_Send(maxVal, BUFF, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(minVal, BUFF, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        MPI_Send(maxComb, BUFF * k, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(minComb, BUFF * k, MPI_INT, 0, 3, MPI_COMM_WORLD);
    }
}