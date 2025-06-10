// @simd
// def fwd_scatter_gather(global_arr : Out[Array[_dfloat]]) -> void:
//         rank : int
//         mpi_rank(rank)
//         nproc : int
//         mpi_size(nproc)
//         init_mpi_env(rank,nproc)
//         chunk : int = mpi_chunk_size()
//         local : Array[_dfloat, 500]
//         scatter(global_arr,local)
//         i : int
//         while (i) < (chunk) :
//                 (local)[i] = make__dfloat((((local)[i]).val) * ((float)(2.0)),((((local)[i]).dval) * ((float)(2.0))) + ((((local)[i]).val) * ((float)(0.0))))
//                 i = (i) + ((int)(1))
//         gather(local,global_arr)

// Generated MPI C code:
#include <mpi.h>
#include <math.h>
#include <stdlib.h>

typedef struct _dfloat {
    float val;
    float dval;
} _dfloat;

void scatter_process_gather(float* global_arr, int total_work);
void fwd_scatter_gather(_dfloat* global_arr, int total_work);
_dfloat make__dfloat(float val, float dval);

void scatter_process_gather(float* global_arr, int _mpi_total_size_s) {
        MPI_Init(NULL, NULL);
        int rank;
        rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int nproc;
        nproc = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);
                int* sendcounts_ = NULL;
        int* displs_ = NULL;
                int mpi_base_ = _mpi_total_size_s / nproc ;
                int mpi_extra_ = _mpi_total_size_s % nproc;
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
        float local[500];
        for (int _i = 0; _i < 500;_i++) {
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

void fwd_scatter_gather(_dfloat* global_arr, int _mpi_total_size_h) {
        MPI_Init(NULL, NULL);
        int rank;
        rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int nproc;
        nproc = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);
                int* sendcounts_ = NULL;
        int* displs_ = NULL;
                int mpi_base_ = _mpi_total_size_h / nproc ;
                int mpi_extra_ = _mpi_total_size_h % nproc;
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
        _dfloat local[500];
        for (int _i = 0; _i < 500;_i++) {
                local[_i].val = 0;
                local[_i].dval = 0;
        }
        MPI_Scatterv(global_arr, sendcounts_, displs_, MPI_FLOAT, local, recvcounts_, MPI_FLOAT, 0, MPI_COMM_WORLD);
        int i;
        i = 0;
        while ((i) < (chunk)) {
                (local)[i] = make__dfloat((((local)[i]).val) * ((float)(2.0)),((((local)[i]).dval) * ((float)(2.0))) + ((((local)[i]).val) * ((float)(0.0))));
                i = (i) + ((int)(1));
        }
        MPI_Gatherv(local, recvcounts_, MPI_FLOAT, global_arr, sendcounts_, displs_, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

_dfloat make__dfloat(float val, float dval) {
        _dfloat ret;
        ret.val = 0;
        ret.dval = 0;
        (ret).val = val;
        (ret).dval = dval;
        return ret;
}