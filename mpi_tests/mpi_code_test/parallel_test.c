#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* 声明外部函数；链接时由 libscatter_process_gather.so 提供 */
extern void scatter_process_gather(float *global_arr, int total_size);

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    /* --------- 测试数据 --------- */
    const int n = 1000;                      /* 必须能被 nproc 整除 */
    if (rank == 0 && n % nproc != 0) {
        fprintf(stderr, "n=%d 不能被进程数 %d 整除\n", n, nproc);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    float *global  = NULL;    /* 只在 root 申请并初始化 */
    float *expect  = NULL;

    if (rank == 0) {
        global = (float *)malloc(n * sizeof(float));
        expect = (float *)malloc(n * sizeof(float));
        if (!global || !expect) {
            fprintf(stderr, "内存不足\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        srand((unsigned)time(NULL));
        for (int i = 0; i < n; ++i) {
            global[i] = (float)rand() / RAND_MAX;
            expect[i] = global[i] * 2.0f;     /* 乘 2 的基准答案 */
        }
    }

    /* --------- 调用共享库函数 --------- */
    scatter_process_gather(global, n);

    MPI_Barrier(MPI_COMM_WORLD);              /* 等所有进程完成 */

    /* --------- 结果校验 --------- */
    if (rank == 0) {
        int ok = 1;
        for (int i = 0; i < n; ++i) {
            if (fabsf(global[i] - expect[i]) > 1e-5f) {
                ok = 0;
                fprintf(stderr,
                        "Mismatch at i=%d : got %f, expect %f\n",
                        i, global[i], expect[i]);
                break;
            }
        }
        printf("%s\n", ok ? "Scatter-process-gather MPI test **passed**!"
                          : "Test **failed**!");
        free(global);
        free(expect);
    }

    MPI_Finalize();
    return 0;
}
