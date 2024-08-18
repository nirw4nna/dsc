from typing import List
import numpy as np


WARMUP = 2
BENCH_STEPS = 5


def random_nd(shape: List[int], dtype: np.dtype = np.float64):
    return np.random.randn(*tuple(shape)).astype(dtype)
