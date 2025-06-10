import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent  = os.path.dirname(current)
sys.path.append(parent)

import compiler

import mpi4py
mpi4py.rc.initialize = False



if __name__ == '__main__':
    print("Compiling MPI code...")
    with open('loma_code/scatter_gather_rev.py', 'r') as f:
        _, lib = compiler.compile(
            f.read(),
            target='mpi',
            output_filename='mpi_code_test/libscatter_gather_rev'  # libscatter_process_gather.so
        )
    print("MPI code compiled successfully.")