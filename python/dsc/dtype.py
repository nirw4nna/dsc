# Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
# All rights reserved.
#
# This code is licensed under the terms of the 3-clause BSD license
# (https://opensource.org/license/bsd-3-clause).

from enum import Enum
import numpy as np
from ctypes import POINTER, c_float, c_double, c_bool, c_int32, c_uint16
from typing import Union


ScalarType = Union[bool, int, float]


class Dtype(Enum):
    BOOL = 0
    I32 = 1
    BF16 = 2
    F32 = 3
    F64 = 4

    def __repr__(self) -> str:
        return TYPENAME_LOOKUP[self]

    def __str__(self) -> str:
        return repr(self)

    @staticmethod
    def from_string(val: str) -> 'Dtype':
        return TYPENAME_REVERSE_LOOKUP[val.lower()]

bool_ = Dtype.BOOL
i32 = Dtype.I32
bf16 = Dtype.BF16
f32 = Dtype.F32
f64 = Dtype.F64

DTYPE_VALUE_LOOKUP = {val.value: val for val in Dtype.__members__.values()}

TYPENAME_LOOKUP = {
    Dtype.BOOL: 'bool',
    Dtype.I32: 'i32',
    Dtype.BF16: 'bf16',
    Dtype.F32: 'f32',
    Dtype.F64: 'f64',
}

TYPENAME_REVERSE_LOOKUP = {v: k for k, v in TYPENAME_LOOKUP.items()}

DTYPE_TO_CTYPE = {
    Dtype.BOOL: POINTER(c_bool),
    Dtype.I32: POINTER(c_int32),
    Dtype.BF16: POINTER(c_uint16),
    Dtype.F32: POINTER(c_float),
    Dtype.F64: POINTER(c_double),
}

DTYPE_SIZE = {
    Dtype.BOOL: 1,
    Dtype.I32: 4,
    Dtype.BF16: 2,
    Dtype.F32: 4,
    Dtype.F64: 8,
}

NP_TO_DTYPE = {
    np.dtype(np.bool): Dtype.BOOL,
    np.dtype(np.int32): Dtype.I32,
    np.dtype(np.float16): Dtype.BF16, # TODO: this is wrong! NumPy doesn't support BF16
    np.dtype(np.float32): Dtype.F32,
    np.dtype(np.float64): Dtype.F64,
}

DTYPE_TO_NP = {val: key for key, val in NP_TO_DTYPE.items()}

DTYPE_CONVERSION_TABLES = [
    [Dtype.BOOL, Dtype.I32, Dtype.F32, Dtype.F32, Dtype.F64],
    [Dtype.BOOL, Dtype.I32, Dtype.F32, Dtype.F32, Dtype.F64],
    [Dtype.F32, Dtype.F32, Dtype.F32, Dtype.F32, Dtype.F64],
    [Dtype.F32, Dtype.F32, Dtype.F32, Dtype.F32, Dtype.F64],
    [Dtype.F64, Dtype.F64, Dtype.F64, Dtype.F64, Dtype.F64],
]
