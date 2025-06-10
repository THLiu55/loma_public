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
            output_filename='loma_code/scatter_process_gather'  # 会生成 libscatter_process_gather.so
        )
    print("MPI code compiled successfully.")