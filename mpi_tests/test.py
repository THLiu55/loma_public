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

import mpi4py
mpi4py.rc.initialize = False

from mpi4py import MPI



class ScatterProcessGatherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        整个测试类只编译一次共享库；多核并发测试时不重复耗时步骤
        """
        # 切到脚本所在目录，方便相对路径
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        MPI.Init()

    def test_scatter_process_gather(self):
        comm       = MPI.COMM_WORLD
        rank       = comm.Get_rank()
        world_size = comm.Get_size()

        # === 1) 生成或加载共享库 ===
        # 如果你已经有 _code/libscatter_process_gather.so，
        # 把下方 compile 注释掉，改成：
        # cls.lib = ctypes.CDLL('_code/libscatter_process_gather.so')
        with open('loma_code/scatter_gather.py', 'r') as f:
            _, self.lib = compiler.compile(
                f.read(),
                target='mpi',
                output_filename='_code/scatter_process_gather'  # 会生成 libscatter_process_gather.so
            )

        # 定义函数原型  void scatter_process_gather(float* global, int total_size)
        self.lib.scatter_process_gather.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # global 数组指针（仅 root 有效）
            ctypes.c_int                     # 元素总数
        ]
        self.lib.scatter_process_gather.restype = None



        # === 2) 准备测试数据 ===
        n =  500
        if n % world_size != 0:
            if rank == 0:
                raise ValueError(f"n={n} 不能被进程数 {world_size} 整除")
            MPI.Finalize()
            return                                    # 其他 rank 提前退出

        if rank == 0:
            global_arr = np.random.rand(n).astype('f')
            expected   = global_arr * 2.0
            global_ptr = global_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            global_arr = None                         # 仅 root 保存结果
            expected   = None
            global_ptr = None                         # 非 root 传 NULL

        # === 3) 调用共享库函数 ===
        self.lib.scatter_process_gather(global_ptr, n)

        # === 4) 断言结果（仅 root）===
        comm.Barrier()
        if rank == 0:
            np.testing.assert_allclose(global_arr, expected, rtol=0, atol=1e-5)
            print("Scatter-process-gather MPI test passed!")

        MPI.Finalize()

if __name__ == '__main__':
    unittest.main()