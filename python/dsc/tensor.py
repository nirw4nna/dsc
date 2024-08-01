from ._bindings import (
    _DscTensor_p, _DSC_MAX_DIMS, _dsc_add, _dsc_sub, _dsc_mul, _dsc_div, _dsc_cast,
    _dsc_addc_f32, _dsc_addc_f64, _dsc_addc_c32, _dsc_addc_c64,
    _dsc_subc_f32, _dsc_subc_f64, _dsc_subc_c32, _dsc_subc_c64,
    _dsc_mulc_f32, _dsc_mulc_f64, _dsc_mulc_c32, _dsc_mulc_c64,
    _dsc_divc_f32, _dsc_divc_f64, _dsc_divc_c32, _dsc_divc_c64,
    _dsc_plan_fft, _dsc_fft, _dsc_ifft, _dsc_arange,
    _dsc_cos, _dsc_sin, _dsc_tensor_1d, _dsc_tensor_2d, _dsc_tensor_3d, _dsc_tensor_4d,
)
from .dtype import *
from .context import _get_ctx
import ctypes
from ctypes import (
    c_uint8,
    c_int
)
import sys


def _c_ptr(x: 'Tensor') -> _DscTensor_p:
    return x._c_ptr if x else None


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

    def _scalar_op(self, other: Union[float, complex], op_base_name: str) -> 'Tensor':
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

        op_name = f'{op_base_name}_{op_dtype}'
        if hasattr(sys.modules[__name__], op_name):
            op = getattr(sys.modules[__name__], op_name)
            return Tensor(op(_get_ctx(), self._c_ptr, other))
        else:
            raise RuntimeError(f'operation "{op_name}" doesn\'t exist in module')

    def __add__(self, other):
        if isinstance(other, (float, complex)):
            return self._scalar_op(other, '_dsc_addc')
        elif isinstance(other, Tensor):
            return Tensor(_dsc_add(_get_ctx(), self._c_ptr, _c_ptr(other)))
        else:
            raise RuntimeError(f'can\'t add Tensor with object of type {type(other)}')

    def __sub__(self, other):
        if isinstance(other, (float, complex)):
            return self._scalar_op(other, '_dsc_subc')
        elif isinstance(other, Tensor):
            return Tensor(_dsc_sub(_get_ctx(), self._c_ptr, _c_ptr(other)))
        else:
            raise RuntimeError(f'can\'t subtract Tensor with object of type {type(other)}')

    def __mul__(self, other: Union[float, complex, 'Tensor']) -> 'Tensor':
        if isinstance(other, (float, complex)):
            return self._scalar_op(other, '_dsc_mulc')
        elif isinstance(other, Tensor):
            return Tensor(_dsc_mul(_get_ctx(), self._c_ptr, _c_ptr(other)))
        else:
            raise RuntimeError(f'can\'t multiply Tensor with object of type {type(other)}')

    def __truediv__(self, other):
        if isinstance(other, (float, complex)):
            return self._scalar_op(other, '_dsc_divc')
        elif isinstance(other, Tensor):
            return Tensor(_dsc_div(_get_ctx(), self._c_ptr, _c_ptr(other)))
        else:
            raise RuntimeError(f'can\'t multiply Tensor with object of type {type(other)}')

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
        return Tensor(_dsc_cast(_get_ctx(), self._c_ptr, c_uint8(dtype.value)))


def from_numpy(x: np.ndarray) -> Tensor:
    if x.dtype not in NP_TO_DTYPE:
        raise RuntimeError(f'NumPy dtype {x.dtype} is not supported')

    dtype = c_uint8(NP_TO_DTYPE[x.dtype].value)

    dims = list(x.shape)
    n_dims = len(dims)
    if n_dims > _DSC_MAX_DIMS or n_dims < 1:
        raise RuntimeError(f'can\'t create a Tensor with {n_dims} dimensions')

    if n_dims == 1:
        res = Tensor(_dsc_tensor_1d(_get_ctx(), dtype, c_int(dims[0])))
    elif n_dims == 2:
        res = Tensor(_dsc_tensor_2d(_get_ctx(), dtype,
                                    c_int(dims[0]), c_int(dims[1])))
    elif n_dims == 3:
        res = Tensor(_dsc_tensor_3d(_get_ctx(), dtype,
                                    c_int(dims[0]), c_int(dims[1]),
                                    c_int(dims[2])))
    else:
        res = Tensor(_dsc_tensor_4d(_get_ctx(), dtype,
                                    c_int(dims[0]), c_int(dims[1]),
                                    c_int(dims[2]), c_int(dims[3])))

    ctypes.memmove(_c_ptr(res).contents.data, x.ctypes.data, x.nbytes)
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

# Todo: implement 'static' functions for add, sub, mul and div so that we can expose also the 'out' parameter
def cos(x: Tensor, out: Tensor = None) -> Tensor:
    return Tensor(_dsc_cos(_get_ctx(), _c_ptr(x), _c_ptr(out)))


def sin(x: Tensor, out: Tensor = None) -> Tensor:
    return Tensor(_dsc_sin(_get_ctx(), _c_ptr(x), _c_ptr(out)))


def arange(n: int, dtype: Dtype = Dtype.F32) -> Tensor:
    return Tensor(_dsc_arange(_get_ctx(), n, c_uint8(dtype.value)))


def plan_fft(n: int, dtype: Dtype = Dtype.F64):
    """
    Create the plan for a one-dimensional FFT/IFFT of size N using dtype for the twiddle factors.
    If this function is not executed before calling either `dsc.fft` or `dsc.ifft` then it will be
    called automatically before doing the first transform causing a slowdown.
    """
    return _dsc_plan_fft(_get_ctx(), n, c_uint8(dtype.value))


def fft(x: Tensor, out: Tensor = None, n: int = -1, axis: int = -1) -> Tensor:
    return Tensor(_dsc_fft(_get_ctx(), _c_ptr(x), _c_ptr(out), n=c_int(n), axis=c_int(axis)))


def ifft(x: Tensor, out: Tensor = None, n: int = -1, axis: int = -1) -> Tensor:
    return Tensor(_dsc_ifft(_get_ctx(), _c_ptr(x), _c_ptr(out), n=c_int(n), axis=c_int(axis)))
