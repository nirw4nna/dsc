from enum import Enum
import numpy as np
from ctypes import (
    POINTER,
    c_float,
    c_double
)
from typing import Union


class Dtype(Enum):
    F32 = 0
    F64 = 1
    C32 = 2
    C64 = 3

    def __repr__(self) -> str:
        return TYPENAME_LOOKUP[self]

    def __str__(self) -> str:
        return repr(self)



TYPENAME_LOOKUP = {
    Dtype.F32: 'f32',
    Dtype.F64: 'f64',
    Dtype.C32: 'c32',
    Dtype.C64: 'c64',
}

DTYPE_TO_CTYPE = {
    Dtype.F32: POINTER(c_float),
    Dtype.F64: POINTER(c_double),
    Dtype.C32: POINTER(c_float * 2),
    Dtype.C64: POINTER(c_double * 2),
}

NP_TO_DTYPE = {
    np.dtype(np.float32): Dtype.F32,
    np.dtype(np.float64): Dtype.F64,
    np.dtype(np.complex64): Dtype.C32,
    np.dtype(np.complex128): Dtype.C64,
}
