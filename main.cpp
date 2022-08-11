#include <bits/stdc++.h>
#include <sys/time.h>
#include "mpi.h"
#include "solver.h"

// dim : dimension of metric space
// k : dimension of rebuilt coord
int n, k, dim;

int main(int argc, char* argv[]) {
    // filename : input file namespace
    char* filename = (char*)"uniformvector-2dim-5h.txt";
    if (argc == 2) {
        filename = argv[1];
    }
    else if (argc != 1) {
        printf("Usage: ./pivot <filename>\n");
        return -1;
    }
    // Read parameters
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("%s file not found.\n", filename);
        return -1;
    }
    fscanf(file, "%d%d%d", &dim, &n, &k);

    // Read Data
    // void* aligned_alloc( std::size_t alignment, std::size_t size );
    double* coord = (double*)aligned_alloc(32, sizeof(double) * dim * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < dim; j++) {
            fscanf(file, "%lf", &coord[i * dim + j]);
        }
    }
    fclose(file);
    // end read

    MPI_Init(&argc, &argv);
    // start timer
    struct timeval start;
    gettimeofday(&start, NULL);

    // begin compute zone
    double* maxVal = (double*)malloc(sizeof(double) * BUFF);
    int* maxComb = (int*)malloc(sizeof(int) * BUFF * k);
    double* minVal = (double*)malloc(sizeof(double) * BUFF);
    int* minComb = (int*)malloc(sizeof(int) * BUFF * k);

    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (k == 2) pivot_solver_2d(n, dim, coord,
        maxVal, maxComb, minVal, minComb, nprocs, rank);
    else pivot_solver_4d(n, dim, coord,
        maxVal, maxComb, minVal, minComb, nprocs, rank);

    commun(k, maxVal, maxComb, minVal, minComb, nprocs, rank);
    // end compute zone

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        struct timeval end;
        gettimeofday(&end, NULL);
        printf("[%d] dim = %d, n = %d, k = %d\n", rank, dim, n, k);
        printf("Using time : %f ms\n",
            (end.tv_sec - start.tv_sec) * 1000.0
            + (end.tv_usec - start.tv_usec) / 1000.0);
        // output
        FILE* out = fopen("result.txt", "w");
        const int M = BUFF;
        for (int i = 0; i < M; i++) {
            int ki;
            for (ki = 0; ki < k - 1; ki++) {
                fprintf(out, "%d ", maxComb[i * k + ki]);
            }
            fprintf(out, "%d\n", maxComb[i * k + k - 1]);
        }
        for (int i = 0; i < M; i++) {
            int ki;
            for (ki = 0; ki < k - 1; ki++) {
                fprintf(out, "%d ", minComb[i * k + ki]);
            }
            fprintf(out, "%d\n", minComb[i * k + k - 1]);
        }
        fclose(out);

        // Log
        int ki;
        printf("max : ");
        for (ki = 0; ki < k; ki++) {
            printf("%d ", maxComb[ki]);
        }
        printf("%lf\n", maxVal[0]);
        printf("min : ");
        for (ki = 0; ki < k; ki++) {
            printf("%d ", minComb[ki]);
        }
        printf("%lf\n", minVal[0]);
    }

    MPI_Finalize();

    return 0;
}
