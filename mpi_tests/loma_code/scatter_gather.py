@simd                             
def scatter_process_gather(global_arr : Out[Array[float]], total_size : In[int]):
    rank  : int
    mpi_rank(rank)
    nproc : int
    mpi_size(nproc)

    chunk : int = total_size / nproc   
    
    init_mpi_env(rank, nproc)

    # 本地缓冲区
    local : Array[float, 5000]

    # ① SCATTER：根进程把 global_arr 拆分给各进程
    # scatter(global_arr, local, total_size)

    # ② 每进程独立处理
    i : int
    while (i < chunk, max_iter := 5000):
        local[i] = local[i] * 2.0

    # ③ GATHER：把处理结果收回根进程
    # gather(local, global_arr, total_size)