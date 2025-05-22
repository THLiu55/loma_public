#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// External add function: returns float
float add(int x, int y) {
    return (float)(x + y);
}

// Perform local vector operation: z[i] = (int)(add(x[i], y[i]))
void simd_local_func(int* x, int* y, int* z, int total_work) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Compute partition
    int base_work = total_work / size;
    int extra = total_work % size;
    int start = base_work * rank + (rank < extra ? rank : extra);
    int end = start + base_work + (rank < extra ? 1 : 0);
    int local_work = end - start;

    // Allocate local arrays
    int* x_local = (int*)malloc(local_work * sizeof(int));
    int* y_local = (int*)malloc(local_work * sizeof(int));
    int* z_local = (int*)malloc(local_work * sizeof(int));

    // Prepare for Scatterv
    int* sendcounts = NULL;
    int* displs = NULL;
    if (rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        for (int i = 0, offset = 0; i < size; ++i) {
            int len = base_work + (i < extra ? 1 : 0);
            sendcounts[i] = len;
            displs[i] = offset;
            offset += len;
        }
    }

    // Scatter data
    MPI_Scatterv(x, sendcounts, displs, MPI_INT, x_local, local_work, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(y, sendcounts, displs, MPI_INT, y_local, local_work, MPI_INT, 0, MPI_COMM_WORLD);

    // Local computation
    for (int i = 0; i < local_work; ++i) {
        z_local[i] = (int)(add(x_local[i], y_local[i]));
    }

    // Gather results
    MPI_Gatherv(z_local, local_work, MPI_INT, z, sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    // Cleanup
    free(x_local);
    free(y_local);
    free(z_local);
    if (rank == 0) {
        free(sendcounts);
        free(displs);
    }
}

// Entry point for MPI version
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int total_work = 1000;
    int* x = NULL;
    int* y = NULL;
    int* z = NULL;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        x = (int*)malloc(total_work * sizeof(int));
        y = (int*)malloc(total_work * sizeof(int));
        z = (int*)malloc(total_work * sizeof(int));
        for (int i = 0; i < total_work; i++) {
            x[i] = i;
            y[i] = i * 2;
        }
    }

    simd_local_func(x, y, z, total_work);

    if (rank == 0) {
        for (int i = 0; i < 10; i++) {
            printf("z[%d] = %d\n", i, z[i]);
        }
        free(x);
        free(y);
        free(z);
    }

    MPI_Finalize();
    return 0;
}
