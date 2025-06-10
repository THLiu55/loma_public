@simd                             
def scatter_process_gather(global_arr : In[Array[float]]):
    rank  : int
    mpi_rank(rank)
    nproc : int
    mpi_size(nproc)
    
    init_mpi_env(rank, nproc)

    chunk : int = mpi_chunk_size()


    # 本地缓冲区
    local : Array[float, 500]

    # ① SCATTER：根进程把 global_arr 拆分给各进程
    scatter(global_arr, local)

    # ② 每进程独立处理
    i : int
    while (i < chunk, max_iter := 500):
        local[i] = local[i] * 2.0
        i = i + 1

    # ③ GATHER：把处理结果收回根进程
    gather(local, global_arr)

rev_scatter_gather = rev_diff(scatter_process_gather)
