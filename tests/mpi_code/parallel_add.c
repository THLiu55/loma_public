#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void parallel_add(int *x, int *y, int *z, int n, int rank, int size) {
    int chunk = n / size;
    int remainder = n % size;

    int start = rank * chunk + (rank < remainder ? rank : remainder);
    int count = chunk + (rank < remainder ? 1 : 0);

    // Local computation
    int *local_z = malloc(count * sizeof(int));
    for (int i = 0; i < count; i++) {
        local_z[i] = x[start + i] + y[start + i];
    }

    // Setup for Gatherv
    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            recvcounts[i] = chunk + (i < remainder ? 1 : 0);
            displs[i] = (i < remainder) ? i * (chunk + 1) : i * chunk + remainder;
        }
    }

    // Gather results to root
    MPI_Gatherv(local_z, count, MPI_INT, z, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    free(local_z);
    if (rank == 0) {
        free(recvcounts);
        free(displs);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = 1000;
    int *x = NULL, *y = NULL, *z = NULL;

    if (rank == 0) {
        x = malloc(n * sizeof(int));
        y = malloc(n * sizeof(int));
        z = malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) {
            x[i] = i;
            y[i] = 2 * i;
        }
    } else {
        x = malloc(n * sizeof(int));
        y = malloc(n * sizeof(int));
    }

    MPI_Bcast(x, n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(y, n, MPI_INT, 0, MPI_COMM_WORLD);

    parallel_add(x, y, z, n, rank, size);

    if (rank == 0) {
        for (int i = 0; i < 10; i++) {
            printf("z[%d] = %d\n", i, z[i]);
        }
        free(z);
    }

    free(x);
    free(y);

    MPI_Finalize();
    return 0;
}