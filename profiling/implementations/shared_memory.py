
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

TILE_SIZE = 16
BLOCK_SIZE = (TILE_SIZE, TILE_SIZE)

@cuda.jit
def rec(u, s, vt, C, k):
    # nice visual explanations: https://siboehm.com/articles/22/CUDA-MMM
    x, y = cuda.grid(2)
    if x >= u.shape[0] or y >= vt.shape[1]:
        return

    thread_x, thread_y = cuda.threadIdx.x, cuda.threadIdx.y

    u_shared = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=float32)
    vt_shared = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=float32)
    s_shared = cuda.shared.array(shape=(TILE_SIZE,), dtype=float32)

    max_blocks = math.ceil(
        k / TILE_SIZE
    )  # or cuda.gridDim.x as we tie the tile size to the block size
    acc = float32(0.0)
    for block_i in range(max_blocks):
        u_shared[thread_x, thread_y] = u[x, block_i * TILE_SIZE + thread_y]
        s_shared[thread_x] = s[block_i * TILE_SIZE + thread_x]
        vt_shared[thread_x, thread_y] = vt[block_i * TILE_SIZE + thread_x, y]

        cuda.syncthreads()

        for n in range(TILE_SIZE):
            acc += u_shared[thread_x, n] * s_shared[n] * vt_shared[n, thread_y]

        cuda.syncthreads()

    C[x, y] = acc


threads_per_block = BLOCK_SIZE
blocks_per_grid = (
    math.ceil(U.shape[0] / BLOCK_SIZE[0]),
    math.ceil(VT.shape[1] / BLOCK_SIZE[1]),
)

# --------------

start_time = cuda.event()
end_time = cuda.event()

start_time.record()
rec[blocks_per_grid, threads_per_block](U_device, S_device, VT_device, C, k)
end_time.record()

cuda.synchronize()
print(f"Done. Execution time: {start_time.elapsed_time(end_time)} ms")
