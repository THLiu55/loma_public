# This file is a python unit test for a scatter-process-gather operation
# Mostly like not runnable or will fail to execute multi-processes

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
        Compile the shared library only once for the entire test class;
        avoids repeated expensive compilation in multi-core test runs.
        """
        # Change to the script directory to simplify relative paths
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        MPI.Init()

    def test_scatter_process_gather(self):
        comm       = MPI.COMM_WORLD
        rank       = comm.Get_rank()
        world_size = comm.Get_size()

        with open('loma_code/scatter_gather.py', 'r') as f:
            _, self.lib = compiler.compile(
                f.read(),
                target='mpi',
                output_filename='_code/scatter_process_gather'  # Will generate libscatter_process_gather.so
            )

        self.lib.scatter_process_gather.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # pointer to global array (valid only on root)
            ctypes.c_int                     # total number of elements
        ]
        self.lib.scatter_process_gather.restype = None

        # Prepare test data
        n = 500
        if n % world_size != 0:
            if rank == 0:
                raise ValueError(f"n={n} cannot be evenly divided by number of processes {world_size}")
            MPI.Finalize()
            return                                    # Other ranks exit early

        if rank == 0:
            global_arr = np.random.rand(n).astype('f')
            expected   = global_arr * 2.0
            global_ptr = global_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            global_arr = None                         
            expected   = None
            global_ptr = None                         

        self.lib.scatter_process_gather(global_ptr, n)

        comm.Barrier()
        if rank == 0:
            np.testing.assert_allclose(global_arr, expected, rtol=0, atol=1e-5)
            print("Scatter-process-gather MPI test passed!")

        MPI.Finalize()

if __name__ == '__main__':
    unittest.main()