from ._bindings import (
    _DscTensor_p, _DSC_MAX_DIMS, _dsc_mul, _dsc_cast, _dsc_mulc_c32, _dsc_mulc_c64, _dsc_mulc_f32, _dsc_mulc_f64,
    _dsc_fft, _dsc_ifft, _dsc_arange, _dsc_cos, _dsc_tensor_1d, _dsc_tensor_2d, _dsc_tensor_3d, _dsc_tensor_4d,
    _dsc_init_fft
)
from .dtype import *
from .context import _get_ctx
import ctypes
from ctypes import (
    c_uint8,
    c_int
)
import sys


class Tensor:
    def __init__(self, c_ptr: _DscTensor_p):
        self._dtype = Dtype(c_ptr.contents.dtype)
        self._shape = c_ptr.contents.shape
        self._n_dim = c_ptr.contents.n_dim
        self._c_ptr = c_ptr

    @property
    def dtype(self) -> Dtype:
        return self._dtype

    @property
    def shape(self) -> tuple[int]:
        return tuple(self._shape[_DSC_MAX_DIMS - self.n_dim:])

    @property
    def n_dim(self) -> int:
        return self._n_dim

    def __mul__(self, other: Union[float, complex, 'Tensor']) -> 'Tensor':
        if isinstance(other, (float, complex)):
            op_dtype = self.dtype
            if isinstance(other, float):
                if self.dtype == Dtype.C32 or self.dtype == Dtype.C64:
                    other = complex(other, 0)
            else:
                # The cast op is handled by the C library
                if self.dtype == Dtype.F32:
                    op_dtype = Dtype.C32
                elif self.dtype == Dtype.F64:
                    op_dtype = Dtype.C64

            op_name = f'_dsc_mulc_{op_dtype}'
            if hasattr(sys.modules[__name__], op_name):
                op = getattr(sys.modules[__name__], op_name)
                return Tensor(op(_get_ctx(), self._c_ptr, other))
            else:
                raise RuntimeError(f'operation "{op_name}" doesn\'t exist in module')
        elif isinstance(other, Tensor):
            return Tensor(_dsc_mul(_get_ctx(), self._c_ptr, other._c_ptr))
        else:
            raise RuntimeError(f'can\'t multiply Tensor with object of type {type(other)}')

    def __add__(self, other):
        # Operators are all very similar, with slightly different names
        pass

    def __sub__(self, other):
        pass

    def __truediv__(self, other):
        pass

    def numpy(self) -> np.ndarray:
        raw_tensor = self._c_ptr.contents

        typed_data = ctypes.cast(raw_tensor.data, DTYPE_TO_CTYPE[self.dtype])

        # Create a view of the underlying data buffer
        np_array = np.ctypeslib.as_array(typed_data, shape=self.shape)

        # If it's a complex number, change the np dtype
        if self.dtype == Dtype.C32:
            np_array = np_array.view(np.complex64)
        elif self.dtype == Dtype.C64:
            np_array = np_array.view(np.complex128)

        return np_array.reshape(self.shape)

    def cast(self, dtype: Dtype) -> 'Tensor':
        return Tensor(_dsc_cast(_get_ctx(), c_uint8(dtype.value), self._c_ptr))


def from_numpy(x: np.ndarray, label: str = '') -> Tensor:
    if x.dtype not in NP_TO_DTYPE:
        raise RuntimeError(f'NumPy dtype {x.dtype} is not supported')

    dtype = c_uint8(NP_TO_DTYPE[x.dtype].value)

    dims = list(x.shape)
    n_dims = len(dims)
    if n_dims > _DSC_MAX_DIMS or n_dims < 1:
        raise RuntimeError(f'can\'t create a Tensor with {n_dims} dimensions')

    if n_dims == 1:
        res = Tensor(_dsc_tensor_1d(_get_ctx(), bytes(label, 'ascii'), dtype,
                                    c_int(dims[0])))
    elif n_dims == 2:
        res = Tensor(_dsc_tensor_2d(_get_ctx(), bytes(label, 'ascii'), dtype,
                                    c_int(dims[0]), c_int(dims[1])))
    elif n_dims == 3:
        res = Tensor(_dsc_tensor_3d(_get_ctx(), bytes(label, 'ascii'), dtype,
                                    c_int(dims[0]), c_int(dims[1]),
                                    c_int(dims[2])))
    else:
        res = Tensor(_dsc_tensor_4d(_get_ctx(), bytes(label, 'ascii'), dtype,
                                    c_int(dims[0]), c_int(dims[1]),
                                    c_int(dims[2]), c_int(dims[3])))

    ctypes.memmove(res._c_ptr.contents.data, x.ctypes.data, x.nbytes)
    return res


# TODO: val can be also an integer, in that case _check_dtype will throw. Handle this case!
#def full(n: int, val: ConstType, dtype: Dtype = Dtype.F32) -> Tensor:
#    _check_dtype(dtype, val)
#
#    if dtype == Dtype.F32:
#        return Tensor(_dsc_full_f32(_get_ctx(), n, val))
#    elif dtype == Dtype.F64:
#        return Tensor(_dsc_full_f64(_get_ctx(), n, val))
#    elif dtype == Dtype.C32:
#        return Tensor(_dsc_full_c32(_get_ctx(), n, val))
#    elif dtype == Dtype.C64:
#        return Tensor(_dsc_full_c64(_get_ctx(), n, val))
#    else:
#        raise RuntimeError(f'Unknown dtype {dtype}')


def cos(x: Tensor) -> Tensor:
    return Tensor(_dsc_cos(_get_ctx(), x._c_ptr))


def arange(n: int, dtype: Dtype = Dtype.F32) -> Tensor:
    return Tensor(_dsc_arange(_get_ctx(), n, c_uint8(dtype.value)))


def plan_fft(n: int, n_workers: int = 1, twiddles: Dtype = Dtype.F64):
    _dsc_init_fft(_get_ctx(), n, n_workers, c_uint8(twiddles.value))


def fft(x: Tensor, axis: int = -1) -> Tensor:
    return Tensor(_dsc_fft(_get_ctx(), x._c_ptr, c_int(axis)))


def ifft(x: Tensor, axis: int = -1) -> Tensor:
    return Tensor(_dsc_ifft(_get_ctx(), x._c_ptr, c_int(axis)))
