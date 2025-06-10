#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* 声明外部函数；链接时由 libscatter_process_gather.so / libscatter_process_gather_rev.so 提供 */
extern void scatter_process_gather(float *global_arr, int total_size);
extern void rev_scatter_gather(float *global_arr, float *dglobal_arr, int total_size);

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    const int n = 1000;  /* 必须能被 nproc 整除 */
    if (n % nproc != 0) {
        if (rank == 0)
            fprintf(stderr, "n=%d 不能被进程数 %d 整除\n", n, nproc);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* 1. 前向测试：global[i] * 2 */
    float *global = NULL, *expect_fwd = NULL;
    if (rank == 0) {
        global    = malloc(n * sizeof(float));
        expect_fwd = malloc(n * sizeof(float));
        srand((unsigned)time(NULL));
        for (int i = 0; i < n; ++i) {
            global[i]     = (float)rand() / RAND_MAX;
            expect_fwd[i] = global[i] * 2.0f;
        }
    }
    /* 调用前向 */
    scatter_process_gather(global, n);
    MPI_Barrier(MPI_COMM_WORLD);

    /* 验证前向结果 */
    if (rank == 0) {
        int ok = 1;
        for (int i = 0; i < n; ++i) {
            if (fabsf(global[i] - expect_fwd[i]) > 1e-5f) {
                ok = 0;
                fprintf(stderr, "Forward Mismatch at %d: got %f, expect %f\n",
                        i, global[i], expect_fwd[i]);
                break;
            }
        }
        printf("%s\n", ok ? "Forward pass passed!" : "Forward pass FAILED!");
        free(expect_fwd);
    }

    /* 2. 逆向测试：对输出 dL/dy 设成 1，看是否能得到 dL/dx = 2 */
    float *dglobal = malloc(n * sizeof(float));
    float *expect_rev = malloc(n * sizeof(float));
    /* 在所有进程上初始化 adjoint 数组，并构造期望值 */
    for (int i = 0; i < n; ++i) {
        dglobal[i]    = 1.0f;        /* 假设 dL/dy = 1 */
        expect_rev[i] = 2.0f;        /* 对 y = 2*x，dL/dx = 2*dL/dy = 2 */
    }

    /* 调用逆向 */
    rev_scatter_gather(global, dglobal, n);
    MPI_Barrier(MPI_COMM_WORLD);

    /* 验证逆向结果 */
    if (rank == 0) {
        int ok = 1;
        for (int i = 0; i < n; ++i) {
            if (fabsf(dglobal[i] - expect_rev[i]) > 1e-5f) {
                ok = 0;
                fprintf(stderr, "Reverse Mismatch at %d: got %f, expect %f\n",
                        i, dglobal[i], expect_rev[i]);
                break;
            }
        }
        printf("%s\n", ok ? "Reverse pass passed!" : "Reverse pass FAILED!");
    }

    free(global);
    free(dglobal);
    free(expect_rev);

    MPI_Finalize();
    return 0;
}
