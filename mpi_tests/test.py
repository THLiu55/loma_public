# file: tests/test_scatter_process_gather_mpi.py
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent  = os.path.dirname(current)
sys.path.append(parent)

import compiler
import ctypes
import unittest
import numpy as np
from mpi4py import MPI

class ScatterProcessGatherTest(unittest.TestCase):
    def setUp(self):
        # 确保工作目录一致，便于找到 _code/ 与 loma_code/
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

    def test_scatter_process_gather_mpi(self):
        # 1) 编译 Loma 源码生成 MPI 动态库
        with open('loma_code/scatter_gather.py') as f:
            structs, lib = compiler.compile(
                f.read(),
                target='mpi',
                output_filename='_code/scatter_process_gather'   # 生成 libscatter_process_gather.so
            )

        # 2) 函数原型：void scatter_process_gather_mpi(float* global, int total_size)
        lib.scatter_process_gather_mpi.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # global_arr（仅 root 有效）
            ctypes.c_int                     # total_size
        ]

        # 3) 测试数据
        n          = 16000                               # 保证能整除进程数
        rank       = MPI.COMM_WORLD.Get_rank()
        world_size = MPI.COMM_WORLD.Get_size()

        if n % world_size != 0 and rank == 0:
            raise ValueError("n 必须能被进程数整除")

        if rank == 0:
            global_arr = np.random.rand(n).astype('f')
            expected   = global_arr * 2.0                # 本地处理逻辑：乘 2
            global_ptr = global_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            global_arr = None
            expected   = None
            global_ptr = None                            # 非 root 传 NULL

        # 4) 调用生成的 MPI 函数
        lib.scatter_process_gather_mpi(global_ptr, n)

        # 5) 同步 & 断言
        MPI.COMM_WORLD.Barrier()
        if rank == 0:
            self.assertTrue(np.allclose(global_arr, expected, atol=1e-6))
            print("Scatter-process-gather MPI test passed!")

if __name__ == '__main__':
    unittest.main()