
from numba import cuda, float32
import math
import os
import pandas as pd
import numpy as np
import sys
import cupy as cp

pwd = os.getcwd()

def load_decompositions(
    filepath=f"{pwd}/data/images.pkl",
):
    if os.path.exists(filepath):
        return pd.read_pickle(filepath)
    return None

decompositions = load_decompositions()
if decompositions is None:
    print("No decompositions found. Run part2.py first.")
    sys.exit(1)

decomposition = max(decompositions.values(), key=lambda d: len(d["s"]))
U, S, VT = decomposition["u"], decomposition["s"], decomposition["vt"]

U_device = cp.array(U, dtype=cp.float32)
S_device = cp.array(S, dtype=cp.float32)
VT_device = cp.array(VT, dtype=cp.float32)

C = np.zeros((U.shape[0], VT.shape[1]), dtype=np.float32)

print(
    f"U shape: {U.shape}, S shape: {S.shape}, VT shape: {VT.shape}, C shape: {C.shape}"
)

k = min(U.shape[1], VT.shape[0]) // 3
print(f"Using k={k}")


# --- Kernel ---

BLOCK_SIZE = (16, 16)

threads_per_block = BLOCK_SIZE


# Updated:
blocks_per_grid = (
    math.ceil(VT.shape[1] / threads_per_block[0]),  # maps to grid idx: x (columns imu)
    math.ceil(U.shape[0] / threads_per_block[1]),  # maps to grid idx: y (rows imu)
)

print(f"Threads per block: {threads_per_block}, Blocks per grid: {blocks_per_grid}")

@cuda.jit
def rec(u, s, vt, C, k):
    # Updated understanding:
    # in the cuda grid we have blocks of threads, each block has a number of threads
    # each thread has a unique identifier (x, y) in the grid
    # x moves along the columns of the matrix, y moves along the rows
    # thus: x is the column index, y is the row index: we map columns to rows, and rows to columns of the output C

    y = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    x = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if x < u.shape[0] and y < vt.shape[1]:
        tmp = float32(0.0)
        for k_i in range(k):
            tmp += u[x, k_i] * s[k_i] * vt[k_i, y]
        C[x, y] = tmp

# --------------

start_time = cuda.event()
end_time = cuda.event()

start_time.record()
rec[blocks_per_grid, threads_per_block](U_device, S_device, VT_device, C, k)
end_time.record()

cuda.synchronize()
print(f"Done. Execution time: {start_time.elapsed_time(end_time)} ms")
