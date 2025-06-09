// test_main_shared.c
#include <stdio.h>
#include <mpi.h>


typedef struct {
    float val;
    float dval;
} _dfloat;



extern void scatter_process_gather(float* global_arr, int total_work);
extern void fwd_scatter_gather(_dfloat* global_arr, int total_work);
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    const int total_size = 10;

    if (rank == 0) {
        float global_arr[total_size];
        for (int i = 0; i < total_size; i++) global_arr[i] = (float)i;

        printf("Before scatter_process_gather: ");
        for (int i = 0; i < total_size; i++) printf("%.1f ", global_arr[i]);
        printf("\n");

        scatter_process_gather(global_arr, total_size);

        printf("After scatter_process_gather:  ");
        for (int i = 0; i < total_size; i++) printf("%.1f ", global_arr[i]);
        printf("\n");

        _dfloat global_dfarr[total_size];
        for (int i = 0; i < total_size; i++) {
            global_dfarr[i].val  = (float)i;
            global_dfarr[i].dval = 1.0f;
        }

        fwd_scatter_gather(global_dfarr, total_size);

        printf("After fwd_scatter_gather:\n");
        for (int i = 0; i < total_size; i++) {
            printf("val: %.1f, dval: %.1f\n",
                   global_dfarr[i].val,
                   global_dfarr[i].dval);
        }
    } else {
        // 非 root 进程也要参与 MPI 调用
        float dummy_arr[1];
        scatter_process_gather(dummy_arr, total_size);

        _dfloat dummy_dfarr[1];
        fwd_scatter_gather(dummy_dfarr, total_size);
    }

    MPI_Finalize();
    return 0;
}
