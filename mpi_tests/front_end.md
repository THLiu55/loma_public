# Check Point of loma OpenMPI Implementation

## Design of Front End
We designed and implemented a set of user-facing interfaces in the Loma language that abstract away low-level MPI details and allow users to write parallel programs with familiar constructs. These functions are automatically lowered to proper MPI API calls during code generation.



Firstly, if the user set target to `mpi`, some arguments will be automatically added, following is one example:
```python
def scatter_process_gather(global_arr : Out[Array[float]]):
```
will be translated like this in c-code:
```c
void scatter_process_gather(float* global_arr, int total_work);
```
Here, the additional int total_work argument represents the **total size of the workload** that needs to be distributed across all MPI processes. This value is critical for partitioning the computation among the processes.



### mpi_rank(_rank : Out[Int])
Assign the rank of the current process to `rank` within MPI_COMM_WORLD.
Usage:

```python
rank_a : int
mpi_rank(rank_a)
```

This funciton will be translated to the following c-code:

```c
int rank_a;
rank_a = 0;
MPI_Comm_rank(MPI_COMM_WORLD, &rank_a);
```

### mpi_size(_nproc : Out[Int])
Returns the total number of processes in MPI_COMM_WORLD.

```python
nproc : int
mpi_size(nproc)
```
This funciton will be translated to the following c-code:

```c
nproc = 0;
MPI_Comm_size(MPI_COMM_WORLD, &nproc);
```

### init_mpi_env(_rank : In[Int], _npoc : In[Int])
> Arguments:
rank: rank of the current process;
size: total number of processes.
    
This function sets up the send counts and displacements needed for MPI_Scatterv and MPI_Gatherv. Internally, it computes the base and extra chunk sizes and broadcasts the scatter/gather meta-info from rank 0 to all other ranks.

Effect: Defines sendcounts, displs, and recvcounts variables used for data distribution.
```python
init_mpi_env(rank, size)
```
This funciton will be translated to the following c-code:
```c
int* sendcounts_ = NULL;
int* displs_ = NULL;
int mpi_base_ = _mpi_total_size_u / nproc ;
int mpi_extra_ = _mpi_total_size_u % nproc;
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
```

Some varibles will be recorded by `OpenMPICodegenVisitor` and be used in the future code generation. Random id can be enabled if we want to make sure there is no variable name confliction.


> Notice: the following function must be called after `init_mpi_env` was called
---

### mpi_chunk_size()
Returns the size of the chunk assigned to the current process, accounting for uneven divisions.
Usage:

```python
chunk : int = mpi_chunk_size()
```
This funciton will be translated to the following c-code:

```clike

int chunk = (recvcounts_);
```

Here the `recvcounts_` is calculated by `init_mpi_env`

### scatter(global_array : In[Array[Int / Float]], local_array : Out[Array[Int / Float]], total_size : In[Int])
Distributes segments of a global array to each process’s local array using MPI_Scatterv.
Usage:

```python
scatter(global_arr, local, total_size);
```

This funciton will be translated to the following c-code:
```c
MPI_Scatterv(global_arr, sendcounts_, displs_, MPI_FLOAT, local, recvcounts_, MPI_FLOAT, 0, MPI_COMM_WORLD);
```

### gather(local_array : In[Array[Int / Float]], global_array  : Out[Array[Int / Float]],  total_size : In[Int])
Gathers local arrays from all ranks into a single global array at rank 0 using MPI_Gatherv.
Usage:

```python
gather(local, global_arr, total_size);
```

This funciton will be translated to the following c-code:

```c
MPI_Gatherv(local, recvcounts_, MPI_FLOAT, global_arr, sendcounts_, displs_, MPI_FLOAT, 0, MPI_COMM_WORLD);
```



## test
**scatter_process_gather**

| Item | Description |
| --- | --- |
| **Goal** | Ensure the shared library `libscatter_process_gather` correctly performs **Scatter → local × 2 → Gather** across multiple MPI ranks. |
| **Test Data** | Rank 0 creates a `float32` array of length `n = 1000`; expected result = input × 2. |
| **Execution Steps** | 1. Initialize MPI<br>2. Load the library and bind the C signature with `ctypes`<br>3. Abort if `n` is not divisible by the number of ranks<br>4. All ranks call `scatter_process_gather`<br>5. `comm.Barrier()` for synchronization<br>6. Rank 0 verifies the output<br>7. Finalize MPI |
| **Pass Criterion** | `np.testing.assert_allclose(actual, expected, atol=1e-5)` passes. |
| **Coverage** | • End-to-end MPI chain<br>• Python ↔ C ↔ MPI interface integrity<br>• Proper synchronization across ranks |

Our code now, with out autodiff, can be successfully pass c-code written test. Here is the code written in loma:

```python=
@simd                             
def scatter_process_gather(global_arr : Out[Array[float]]):
    rank  : int
    mpi_rank(rank)
    nproc : int
    mpi_size(nproc)
    
    init_mpi_env(rank, nproc)

    chunk : int = mpi_chunk_size()


    local : Array[float, 500]

    scatter(global_arr, local)

    i : int
    while (i < chunk, max_iter := 500):
        local[i] = local[i] * 2.0
        i = i + 1

    gather(local, global_arr)
```

This will be translated to c-code like this:
```c=
// Generated MPI C code:
#include <mpi.h>
#include <math.h>
#include <stdlib.h>

void scatter_process_gather(float* global_arr, int total_work);

void scatter_process_gather(float* global_arr, int _mpi_total_size_u) {
        int rank;
        rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int nproc;
        nproc = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);
        int* sendcounts_ = NULL;
        int* displs_ = NULL;
        int mpi_base_ = _mpi_total_size_u / nproc ;
        int mpi_extra_ = _mpi_total_size_u % nproc;
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
```