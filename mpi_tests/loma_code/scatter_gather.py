@simd                             
def scatter_process_gather(global_arr : InOut[Array[float]], total_size : In[int]) -> void:
    rank  : int = mpi_rank()
    nproc : int = mpi_size()
    chunk : int = total_size / nproc          # 每段长度

    # 本地缓冲区
    local : Array[float, dynamic] = alloc_array_float(chunk)

    # ① SCATTER：根进程把 global_arr 拆分给各进程
    scatter(global_arr, local, total_size)

    # ② 每进程独立处理
    i : int
    for i in range(chunk):
        local[i] = local[i] * 2.0

    # ③ GATHER：把处理结果收回根进程
    gather(local, global_arr, total_size)