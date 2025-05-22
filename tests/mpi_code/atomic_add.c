#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void my_atomic_add(float *x, int n, float *z, int rank, int size) {
    int chunk = n / size;
    int start = rank * chunk;
    int end = (rank == size - 1) ? n : start + chunk;

    float local_sum = 0.0;
    for (int i = start; i < end; i++) {
        local_sum += x[i];
    }

    // Reduce all local sums to global sum in process 0
    MPI_Reduce(&local_sum, z, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = 1000;
    float *x = NULL;
    float z = 0.0;

    if (rank == 0) {
        x = (float *)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) x[i] = 1.0f;  // example input
    }

    // Broadcast x to all processes
    if (rank != 0) x = (float *)malloc(n * sizeof(float));
    MPI_Bcast(x, n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    my_atomic_add(x, n, &z, rank, size);

    if (rank == 0) {
        printf("Final sum: %f\n", z);
        free(x);
    } else {
        free(x);
    }

    MPI_Finalize();
    return 0;
}