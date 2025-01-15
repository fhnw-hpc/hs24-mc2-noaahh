
import math
import time
import os

import threading
from queue import Queue, Empty

from dataclasses import dataclass
from typing import List, Dict
import logging

import numpy as np
from argparse import ArgumentParser
from numba import cuda, float32
from numpy.linalg import svd
from PIL import Image

# --- Constants ---

LOG_LEVEL = logging.INFO
OUTPUT_DIR = "out"

BLOCK_SIZE_X, BLOCK_SIZE_Y = 16, 16
TOTAL_IMAGES = 10000
IMG_SIZE = 2**10

NUM_WORKERS = 10
NUM_STREAMS = 6

# ------------------

IMG = np.random.rand(IMG_SIZE, IMG_SIZE).astype(np.float32)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


@cuda.jit
def reconstruct(u, s, vt, C, k):
    y = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    x = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if x < u.shape[0] and y < vt.shape[1]:
        tmp = float32(0.0)
        for k_i in range(k):
            tmp += u[x, k_i] * s[k_i] * vt[k_i, y]
        C[x, y] = tmp


def save_to_disk(data, task_id):
    """Save the reconstructed result to disk as a PNG file."""
    # early return for testing but we could theoretically save the image to the disk or do whatever we want
    logger.debug(f"Processing task {task_id} result in callback")
    return

    is_close = np.allclose(data, IMG, atol=0.1)
    if not is_close:
        logger.error(
            f"Task {task_id} result is not close to the original image. Max diff: {np.max(np.abs(data - IMG))}"
        )

    image = Image.fromarray((data * 255).astype(np.uint8))
    image.save(f"out/reconstructed_{task_id}.png")
    logger.info(f"Saved task {task_id} result to disk.")


@dataclass
class GPUResources:

    streams: List[cuda.stream]
    device_arrays: List[Dict[str, cuda.devicearray.DeviceNDArray]]
    pinned_arrays: List[Dict[str, np.ndarray]]


class GPUProcessor:
    def __init__(self, shape_info: tuple):
        self.num_workers = NUM_WORKERS
        self.num_streams = NUM_STREAMS
        self.work_queue = Queue()
        self.shutdown_event = threading.Event()

        self.workers: List[threading.Thread] = []
        self.worker_resources: List[GPUResources] = []

        for i in range(NUM_WORKERS):
            logger.info(f"Initializing resources for worker {i}")
            self.worker_resources.append(self._initialize_gpu_resources(shape_info))
            logger.info(f"Resources initialized for worker {i}")

    def _initialize_gpu_resources(self, shape_info: tuple) -> GPUResources:
        """Initialize all GPU resources on the host thread"""
        M, N, K = shape_info

        streams = [cuda.stream() for _ in range(self.num_streams)]
        device_arrays = []
        pinned_arrays = []

        for _ in range(self.num_streams):
            d_arrays = {
                "U": cuda.device_array((M, K), dtype=np.float32),
                "S": cuda.device_array((K,), dtype=np.float32),
                "V": cuda.device_array((K, N), dtype=np.float32),
                "C": cuda.device_array((M, N), dtype=np.float32),
            }
            device_arrays.append(d_arrays)

            p_arrays = {
                "U": cuda.pinned_array((M, K), dtype=np.float32),
                "S": cuda.pinned_array((K,), dtype=np.float32),
                "V": cuda.pinned_array((K, N), dtype=np.float32),
                "C": cuda.pinned_array((M, N), dtype=np.float32),
            }
            pinned_arrays.append(p_arrays)

        return GPUResources(streams, device_arrays, pinned_arrays)

    def gpu_worker(self, worker_id: int, shape_info: tuple):
        """GPU worker thread that processes items from the work queue"""
        M, N, K = shape_info
        resources = self.worker_resources[worker_id]

        def stream_callback(stream, status, arg):
            task_id, slot_id = arg
            # do something with the result
            #result_data = resources.pinned_arrays[slot_id]["C"]
            #save_to_disk(result_data, task_id)

        logger.info(f"Worker {worker_id} started")
        counter = 0
        try:
            while not self.shutdown_event.is_set():
                try:
                    task = self.work_queue.get(timeout=1.0)
                    task_id, decomp = task

                    slot_id = counter % self.num_streams
                    counter += 1

                    stream = resources.streams[slot_id]
                    d_arrays = resources.device_arrays[slot_id]
                    p_arrays = resources.pinned_arrays[slot_id]

                    p_arrays["U"][:] = decomp["u"]
                    p_arrays["S"][:] = decomp["s"]
                    p_arrays["V"][:] = decomp["v"]
                    cuda.to_device(p_arrays["U"], to=d_arrays["U"], stream=stream)
                    cuda.to_device(p_arrays["S"], to=d_arrays["S"], stream=stream)
                    cuda.to_device(p_arrays["V"], to=d_arrays["V"], stream=stream)

                    grid_x = math.ceil(N / BLOCK_SIZE_X)
                    grid_y = math.ceil(M / BLOCK_SIZE_Y)
                    reconstruct[(grid_y, grid_x), (BLOCK_SIZE_X, BLOCK_SIZE_Y), stream](
                        d_arrays["U"], d_arrays["S"], d_arrays["V"], d_arrays["C"], K
                    )

                    d_arrays["C"].copy_to_host(p_arrays["C"], stream=stream)
                    stream.add_callback(stream_callback, (task_id, slot_id))

                    self.work_queue.task_done()
                    logger.debug(
                        f"Worker {worker_id} - completed task {task_id} - slot {slot_id} - counter {counter}"
                    )

                except Empty:
                    logger.info(f"Worker {worker_id} is idling...")
                    continue

                except Exception as e:
                    logger.error(
                        f"Error in worker {worker_id}: {str(e)}", exc_info=True
                    )

        except Exception as e:
            logger.error(f"Fatal error in worker {worker_id}: {str(e)}", exc_info=True)
        finally:
            logger.info(f"Worker {worker_id} shutting down...")

            logger.info(f"Worker {worker_id} cleaning up resources...")
            for stream in resources.streams:
                stream.synchronize()
            logger.info(f"Worker {worker_id} resources cleaned up")

            logger.info(f"Worker {worker_id} shutdown successfully")

    def start_processing(self, shape_info: tuple):
        for worker_id in range(self.num_workers):
            worker = threading.Thread(
                target=self.gpu_worker, args=(worker_id, shape_info)
            )
            worker.start()
            self.workers.append(worker)

    def add_work(self, decomp: Dict, task_id: int):
        self.work_queue.put((task_id, decomp))

    def stop(self):
        logger.info("Initiating shutdown...")
        self.shutdown_event.set()
        self.work_queue.join()

        for worker in self.workers:
            worker.join()

        logger.info("All workers shut down successfully")


def process_sequential(shape_info, decomp):
    M, N, K = shape_info

    U_pinned = cuda.pinned_array((M, K), dtype=np.float32)
    S_pinned = cuda.pinned_array((K,), dtype=np.float32)
    V_pinned = cuda.pinned_array((K, N), dtype=np.float32)
    C_pinned = cuda.pinned_array((M, N), dtype=np.float32)

    U_device = cuda.device_array((M, K), dtype=np.float32)
    S_device = cuda.device_array((K,), dtype=np.float32)
    V_device = cuda.device_array((K, N), dtype=np.float32)
    C_device = cuda.device_array((M, N), dtype=np.float32)

    start_time = time.time()
    for i in range(TOTAL_IMAGES):
        U_pinned[:] = decomp["u"]
        S_pinned[:] = decomp["s"]
        V_pinned[:] = decomp["v"]

        cuda.to_device(U_pinned, to=U_device)
        cuda.to_device(S_pinned, to=S_device)
        cuda.to_device(V_pinned, to=V_device)

        grid_x = math.ceil(N / BLOCK_SIZE_X)
        grid_y = math.ceil(M / BLOCK_SIZE_Y)
        reconstruct[(grid_y, grid_x), (BLOCK_SIZE_X, BLOCK_SIZE_Y)](
            U_device, S_device, V_device, C_device, K
        )

        C_device.copy_to_host(C_pinned)

    cuda.synchronize()

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(
        f"Total time: {total_time:.2f}s, images / s: {TOTAL_IMAGES/total_time:.2f}, images: {TOTAL_IMAGES}"
    )


def main():
    logger.info(f"Num workers: {NUM_WORKERS}, Num streams / worker: {NUM_STREAMS}")

    parser = ArgumentParser()
    parser.add_argument(
        "--sequential", action="store_true", help="Run the computation sequentially"
    )
    args = parser.parse_args()
    sequential = args.sequential
    logger.info(f"Sequential: {sequential}")

    u, v, t = svd(IMG, full_matrices=False)
    decomp = {"u": u, "s": v, "v": t}

    M, K = decomp["u"].shape
    K_check, N = decomp["v"].shape

    if K_check != K:
        raise ValueError("Mismatch between U and V shapes in SVD.")

    shape_info = (M, N, K)
    logger.info(f"Shape info: {shape_info}")

    if sequential:
        process_sequential(shape_info, decomp)
        return

    start_time = time.time()
    processor = GPUProcessor(shape_info)
    try:
        for i in range(TOTAL_IMAGES):
            processor.add_work(decomp, i)

        processor.start_processing(shape_info)

        logger.info("Waiting for empty queue...")
        processor.work_queue.join()
        logger.info("Queue is empty")

    except Exception as e:
        logger.error(f"Error in main processing loop: {str(e)}", exc_info=True)
    finally:
        logger.info("Shutting down processor...")
        processor.stop()
        logger.info("Processor shutdown successfull")

    end_time = time.time()
    logger.info(
        f"Total time taken: {end_time - start_time:.2f} second - images / second: {TOTAL_IMAGES / (end_time - start_time):.2f}, images: {TOTAL_IMAGES}"
    )


if __name__ == "__main__":
    main()
