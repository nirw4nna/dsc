# Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
# All rights reserved.
#
# This code is licensed under the terms of the 3-clause BSD license
# (https://opensource.org/license/bsd-3-clause).


import dsc
import numpy as np
from typing import List
import os


DEVICE = os.getenv('DEVICE', 'cpu')

def all_close(actual: dsc.Tensor, target: np.ndarray, eps=1e-5):
    actual_np = actual.numpy()
    diffs = ~np.isclose(actual_np, target, atol=eps, rtol=eps, equal_nan=True)
    close = len(actual_np[diffs]) == 0
    return close


def random_nd(shape: List[int], dtype: np.dtype = np.float64):
    if dtype == np.bool:
        return np.random.randint(0, 2, size=tuple(shape)).astype(dtype)
    elif dtype == np.int32:
        # Return a positive integer tensor if the dtype is int32 so that we don't have issues
        # with power
        return np.random.randint(0, 10, size=tuple(shape)).astype(dtype)
    else:
        return np.random.randn(*tuple(shape)).astype(dtype)


DTYPES = [np.bool, np.int32, np.float32, np.float64]
DSC_DTYPES = {
    np.bool: dsc.bool_,
    np.int32: dsc.i32,
    np.float32: dsc.f32,
    np.float64: dsc.f64,
}

def is_float(dtype) -> bool:
    return dtype == np.float32 or dtype == np.float64

def is_bool(dtype) -> bool:
    return dtype == np.bool

def is_integer(dtype) -> bool:
    return dtype == np.int32
