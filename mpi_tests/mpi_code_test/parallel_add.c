#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>


void scatter_process_gather(float* global_arr, int total_size) {
        int rank;
        rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int nproc;
        nproc = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);
        int chunk = (total_size) / (nproc);
                int* sendcounts = NULL;
        int* displs = NULL;
        if (rank == 0) {
                sendcounts = malloc(nproc * sizeof(int));
                displs     = malloc(nproc * sizeof(int));
                int base = total_size / nproc;
                int extra = total_size % nproc;
                int offset = 0;
                for (int i = 0; i < nproc; i++) {
                    sendcounts[i] = base + (i < extra ? 1 : 0);
                    displs[i] = offset;
                    offset += sendcounts[i];
                }
            };
        float local[5000];
        for (int _i = 0; _i < 5000;_i++) {
                local[_i] = 0;
        }
        int i;
        i = 0;
        while ((i) < (chunk)) {
                (local)[i] = ((local)[i]) * ((float)(2.0));
                i = (i) + ((int)(1));
        }
}

int main(int argc, char** argv) {
    int total_size = 16;
    float* data = NULL;

    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        data = malloc(sizeof(float) * total_size);
        for (int i = 0; i < total_size; i++) {
            data[i] = (float)(i + 1);  // e.g., [1.0, 2.0, ..., 16.0]
        }
    }

    scatter_process_gather(data, total_size);

    if (rank == 0) {
        free(data);
    }

    MPI_Finalize();
    return 0;
}
