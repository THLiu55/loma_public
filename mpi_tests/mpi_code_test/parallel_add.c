#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>


void scatter_process_gather(float* global_arr, int _mpi_total_size_r) {
        int rank;
        rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int nproc;
        nproc = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);
                int* sendcounts_ = NULL;
        int* displs_ = NULL;
                int mpi_base_ = _mpi_total_size_r / nproc ;
                int mpi_extra_ = _mpi_total_size_r % nproc;
        int recvcounts_ =  mpi_base_ + (rank < mpi_extra_  ? 1 : 0);
        if (rank == 0) {
                sendcounts_ = malloc(nproc * sizeof(int));
                 displs_      = malloc(nproc * sizeof(int));
                int offset = 0;
                for (int i = 0; i < nproc; i++) {
                    sendcounts_[i] = mpi_base_ + (i < mpi_extra_  ? 1 : 0);
                     displs_[i] = offset;
                    offset += sendcounts_[i];
                }
            };
        int chunk = (recvcounts_);
        float local[5000];
        for (int _i = 0; _i < 5000;_i++) {
                local[_i] = 0;
        }
        MPI_Scatterv(global_arr, sendcounts_, displs_, MPI_FLOAT, local, recvcounts_, MPI_FLOAT, 0, MPI_COMM_WORLD);
        int i;
        i = 0;
        while ((i) < (chunk)) {
                (local)[i] = ((local)[i]) * ((float)(2.0));
                i = (i) + ((int)(1));
        }
        MPI_Gatherv(local, recvcounts_, MPI_FLOAT, global_arr, sendcounts_, displs_, MPI_FLOAT, 0, MPI_COMM_WORLD);
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
        printf("Processed data: ");
        for (int i = 0; i < total_size; i++) {
            printf("%.1f ", data[i]);  // Expecting [2.0, 4.0, ..., 32.0]
        }
        printf("\n");
    }

    if (rank == 0) {
        free(data);
    }

    MPI_Finalize();
    return 0;
}
