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

export void rev_parallel_add_mpi(float* x,
                                 float* _dx_,
                                 float* y,
                                 float* _dy_,
                                 float* z,
                                 int    total_work)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ---------- 与 forward 保持一致的分区 ---------- */
    int base_work = total_work / size;
    int extra     = total_work % size;                 /* 把余数分到前 extra 个进程 */
    int start     = base_work * rank + (rank < extra ? rank : extra);
    int end       = start + base_work + (rank < extra ? 1 : 0);
    int local_work = end - start;

    /* ---------- 根进程提前准备 Scatterv/Gatherv 参数 ---------- */
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

    /* ---------- 为本地片段分配缓冲 ---------- */
    float *z_local  = (float*)malloc(local_work * sizeof(float));
    float *dx_local = (float*)calloc(local_work, sizeof(float));   /* 初值 0 */
    float *dy_local = (float*)calloc(local_work, sizeof(float));

    /* ---------- 把 z（正向结果）分发到各 rank ---------- */
    MPI_Scatterv(z,  sendcounts, displs, MPI_FLOAT,
                 z_local,  local_work, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    /* 如果想在反向阶段继续累加已有梯度，可把 _dx_/ _dy_ 也 Scatter 下来。
       若梯度在此函数之前尚未被写入，一般保持 0 即可： */
    // MPI_Scatterv(_dx_, sendcounts, displs, MPI_FLOAT,
    //              dx_local, local_work, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // MPI_Scatterv(_dy_, sendcounts, displs, MPI_FLOAT,
    //              dy_local, local_work, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* ---------- 反向本地计算：dx += z, dy += z ---------- */
    for (int i = 0; i < local_work; ++i) {
        dx_local[i] += z_local[i];
        dy_local[i] += z_local[i];
    }

    /* ---------- 把局部梯度收集回根进程 ---------- */
    MPI_Gatherv(dx_local, local_work, MPI_FLOAT,
                _dx_, sendcounts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    MPI_Gatherv(dy_local, local_work, MPI_FLOAT,
                _dy_, sendcounts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    /* ---------- 清理 ---------- */
    free(z_local);
    free(dx_local);
    free(dy_local);
    if (rank == 0) {
        free(sendcounts);
        free(displs);
    }
}
