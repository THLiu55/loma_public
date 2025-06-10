#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

extern void scatter_process_gather(float *global_arr, int total_size);

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // Test data
    const int n = 1000;                      
    if (rank == 0 && n % nproc != 0) {
        fprintf(stderr, "n=%d cannot be evenly divided by number of processes %d\n", n, nproc);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    float *global  = NULL;    /* Allocate and initialize only on root */
    float *expect  = NULL;

    if (rank == 0) {
        global = (float *)malloc(n * sizeof(float));
        expect = (float *)malloc(n * sizeof(float));
        if (!global || !expect) {
            fprintf(stderr, "Out of memory\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        srand((unsigned)time(NULL));
        for (int i = 0; i < n; ++i) {
            global[i] = (float)rand() / RAND_MAX;
            expect[i] = global[i] * 2.0f;     /* Reference answer: multiply by 2 */
        }
    }

    // Call
    scatter_process_gather(global, n);

    MPI_Barrier(MPI_COMM_WORLD);              /* Wait for all processes to finish */

    // Verify result
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
