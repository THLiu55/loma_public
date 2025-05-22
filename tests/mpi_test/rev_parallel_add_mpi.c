#include <mpi.h>
#include <stdlib.h>

void rev_parallel_add_mpi(float* x,
                          float* _dx_,
                          float* y,
                          float* _dy_,
                          float* z,
                          int    total_work)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int base_work = total_work / size;
    int extra     = total_work % size;
    int start     = base_work * rank + (rank < extra ? rank : extra);
    int end       = start + base_work + (rank < extra ? 1 : 0);
    int local_work = end - start;

    int *sendcounts = NULL, *displs = NULL;
    if (rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs     = (int*)malloc(size * sizeof(int));
        for (int i = 0, offset = 0; i < size; ++i) {
            int len = base_work + (i < extra ? 1 : 0);
            sendcounts[i] = len;
            displs[i]     = offset;
            offset       += len;
        }
    }

    float *z_local  = (float*)malloc(local_work * sizeof(float));
    float *dx_local = (float*)calloc(local_work, sizeof(float));
    float *dy_local = (float*)calloc(local_work, sizeof(float));

    MPI_Scatterv(z, sendcounts, displs, MPI_FLOAT,
                 z_local, local_work, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_work; ++i) {
        dx_local[i] += z_local[i];
        dy_local[i] += z_local[i];
    }

    MPI_Gatherv(dx_local, local_work, MPI_FLOAT,
                _dx_, sendcounts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    MPI_Gatherv(dy_local, local_work, MPI_FLOAT,
                _dy_, sendcounts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    free(z_local);
    free(dx_local);
    free(dy_local);
    if (rank == 0) {
        free(sendcounts);
        free(displs);
    }
}