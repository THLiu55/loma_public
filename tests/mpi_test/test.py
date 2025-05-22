import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)
import compiler
import ctypes
import error
import math
import gpuctypes.opencl as cl
import cl_utils
import unittest
import numpy as np
from mpi4py import MPI

epsilon = 1e-4

class Homework3Test(unittest.TestCase):
    def setUp(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))


    def test_parallel_add_mpi(self):
        lib = ctypes.CDLL("./librev_add.so")
        lib.rev_parallel_add_mpi.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # x
            ctypes.POINTER(ctypes.c_float),  # _dx_
            ctypes.POINTER(ctypes.c_float),  # y
            ctypes.POINTER(ctypes.c_float),  # _dy_
            ctypes.POINTER(ctypes.c_float),  # z
            ctypes.c_int                     # total_work
        ]

        np.random.seed(seed=1234)
        n = 10000
        x = (np.random.rand(n).astype('f') / n)
        y = (np.random.rand(n).astype('f') / n)
        z = np.zeros_like(x)
        _dz = (np.random.rand(n).astype('f') / n)
        _dx = np.zeros_like(x)
        _dy = np.zeros_like(y)

        # 只根进程负责传 z，其它进程传 None 以避免非法内存
        z_ptr = _dz.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if MPI.COMM_WORLD.Get_rank() == 0 else None
        _dx_ptr = _dx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if MPI.COMM_WORLD.Get_rank() == 0 else None
        _dy_ptr = _dy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if MPI.COMM_WORLD.Get_rank() == 0 else None

        lib.rev_parallel_add_mpi(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            _dx_ptr,
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            _dy_ptr,
            z_ptr,
            n
        )

        # 验证
        if MPI.COMM_WORLD.Get_rank() == 0:
            epsilon = 1e-6
            assert np.allclose(_dx, _dz, atol=epsilon)
            assert np.allclose(_dy, _dz, atol=epsilon)
            print("Test passed!")



if __name__ == '__main__':
    unittest.main()