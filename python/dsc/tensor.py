# Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
# All rights reserved.
#
# This code is licensed under the terms of the 3-clause BSD license
# (https://opensource.org/license/bsd-3-clause).

from ._bindings import (
    _DscTensor_p,
    _OptionalTensor,
    _DSC_MAX_DIMS,
    _DSC_SLICE_NONE,
    _DscSlice,
    _dsc_cast,
    _dsc_reshape,
    _dsc_tensor_free,
    _dsc_sum,
    _dsc_mean,
    _dsc_max,
    _dsc_min,
    _dsc_i0,
    _dsc_clip,
    _dsc_tensor_get_idx,
    _dsc_tensor_get_slice,
    _dsc_tensor_set_idx,
    _dsc_tensor_set_slice,
    _dsc_wrap_f32,
    _dsc_wrap_f64,
    _dsc_wrap_c32,
    _dsc_wrap_c64,
    _dsc_add,
    _dsc_sub,
    _dsc_mul,
    _dsc_div,
    _dsc_pow,
    _dsc_plan_fft,
    _dsc_fft,
    _dsc_ifft,
    _dsc_rfft,
    _dsc_irfft,
    _dsc_fftfreq,
    _dsc_rfftfreq,
    _dsc_arange,
    _dsc_randn,
    _dsc_cos,
    _dsc_sin,
    _dsc_sinc,
    _dsc_logn,
    _dsc_log2,
    _dsc_log10,
    _dsc_exp,
    _dsc_sqrt,
    _dsc_abs,
    _dsc_angle,
    _dsc_conj,
    _dsc_real,
    _dsc_imag,
    _dsc_tensor_1d,
    _dsc_tensor_2d,
    _dsc_tensor_3d,
    _dsc_tensor_4d,
)
from .dtype import (
    Dtype,
    NP_TO_DTYPE,
    DTYPE_CONVERSION_TABLES,
    DTYPE_SIZE,
    DTYPE_TO_CTYPE,
    ScalarType,
)
import numpy as np
from .context import _get_ctx
import ctypes
import sys
from typing import Union, Tuple, List


TensorType = Union['Tensor', np.ndarray]


def _c_ptr_or_none(x: Union['Tensor', None]) -> _OptionalTensor:
    return x._c_ptr if x else None


def _c_ptr(x: 'Tensor') -> _DscTensor_p:
    return x._c_ptr


def _unwrap(x: 'Tensor') -> Union[float, complex, 'Tensor']:
    # If x is not wrapping a single value return it
    if x.n_dim != 1 or len(x) != 1:
        return x

    x_ptr = x._c_ptr.contents.data
    if x.dtype == Dtype.F32 or x.dtype == Dtype.F64:
        return ctypes.cast(x_ptr, DTYPE_TO_CTYPE[x.dtype]).contents.value
    elif x.dtype == Dtype.C32 or x.dtype == Dtype.C64:
        complex_arr = ctypes.cast(x_ptr, DTYPE_TO_CTYPE[x.dtype]).contents
        return complex(complex_arr[0], complex_arr[1])
    else:
        raise RuntimeError(f'unknown dtype {x.dtype}')


def _c_slice(x: Union[slice, int]) -> _DscSlice:
    def _sanitize(i: Union[int, None]) -> int:
        if i is None:
            return _DSC_SLICE_NONE
        return i

    if isinstance(x, slice):
        return _DscSlice(_sanitize(x.start), _sanitize(x.stop), _sanitize(x.step))
    else:
        # This is needed for when indexes and slices are mixed: suppose x[-1, ::-2] since we don't have a method that
        # takes both indexes and slices this will generate a new slice (-1) where start=dim stop=dim step=dim. All the
        # handling will be done internally by the library.
        return _DscSlice(x, x, x)


def _wrap(
    x: Union[ScalarType, 'Tensor', np.ndarray], dtype: Union[Dtype, None] = None
) -> 'Tensor':
    # Default dtype is f32 or c32, depending on float or complex
    if isinstance(x, np.ndarray):
        # The conversion from (and to) NumPy is not free, so it's better to do that once and then work with DSC tensors.
        # This is here just for convenience not because it's a best practice.
        return from_numpy(x)
    elif isinstance(x, Tensor):
        return x
    elif isinstance(x, float):
        if dtype == Dtype.F64:
            return Tensor(_dsc_wrap_f64(_get_ctx(), x))
        else:
            return Tensor(_dsc_wrap_f32(_get_ctx(), x))
    elif isinstance(x, complex):
        if dtype == Dtype.C64:
            return Tensor(_dsc_wrap_c64(_get_ctx(), x))
        else:
            return Tensor(_dsc_wrap_c32(_get_ctx(), x))
    else:
        # Simply cast x to dtype, if any
        if dtype == Dtype.F32 or dtype is None:
            return Tensor(_dsc_wrap_f32(_get_ctx(), float(x)))
        elif dtype == Dtype.F64:
            return Tensor(_dsc_wrap_f64(_get_ctx(), float(x)))
        elif dtype == Dtype.C32:
            return Tensor(_dsc_wrap_c32(_get_ctx(), complex(x, 0)))
        elif dtype == Dtype.C64:
            return Tensor(_dsc_wrap_c64(_get_ctx(), complex(x, 0)))


class Tensor:
    def __init__(self, c_ptr: _DscTensor_p):  # pyright: ignore[reportInvalidTypeForm]
        self._dtype = Dtype(c_ptr.contents.dtype)
        self._shape = c_ptr.contents.shape
        self._n_dim = c_ptr.contents.n_dim
        self._ne = c_ptr.contents.ne
        self._c_ptr = c_ptr

    def __del__(self):
        _dsc_tensor_free(_get_ctx(), _c_ptr(self))

    @property
    def dtype(self) -> Dtype:
        return self._dtype

    @property
    def shape(self) -> tuple[int]:
        return tuple(self._shape[_DSC_MAX_DIMS - self.n_dim :])

    @property
    def n_dim(self) -> int:
        return self._n_dim

    @property
    def ne(self) -> int:
        return self._ne

    def __len__(self):
        return self.shape[0]

    def __str__(self) -> str:
        return str(self.numpy())

    def __getitem__(
        self,
        item: Union[
            int,
            Tuple[int, ...],
            slice,
            Tuple[slice, ...],
            Tuple[Union[int, slice], ...],
        ],
    ) -> Union[float, complex, 'Tensor']:
        if isinstance(item, int):
            return _unwrap(Tensor(_dsc_tensor_get_idx(_get_ctx(), _c_ptr(self), item)))
        elif isinstance(item, Tuple) and all(isinstance(i, int) for i in item):
            return _unwrap(
                Tensor(
                    _dsc_tensor_get_idx(
                        _get_ctx(),
                        _c_ptr(self),
                        *tuple(i for i in item),  # pyright: ignore[reportArgumentType]
                    )
                )
            )
        elif isinstance(item, slice):
            return Tensor(
                _dsc_tensor_get_slice(_get_ctx(), _c_ptr(self), _c_slice(item))
            )
        elif (isinstance(item, Tuple) and all(isinstance(s, slice) for s in item)) or (
            isinstance(item, Tuple)
            and all(isinstance(i, int) or isinstance(i, slice) for i in item)
        ):
            return Tensor(
                _dsc_tensor_get_slice(
                    _get_ctx(), _c_ptr(self), *tuple(_c_slice(s) for s in item)
                )
            )
        else:
            raise RuntimeError(f'cannot index Tensor with object {item}')

    def __setitem__(
        self,
        key: Union[
            int,
            Tuple[int, ...],
            slice,
            Tuple[slice, ...],
            Tuple[Union[int, slice], ...],
        ],
        value: Union[ScalarType, 'Tensor', np.ndarray],
    ):
        wrapped_val = _wrap(value, self.dtype)
        if isinstance(key, int):
            _dsc_tensor_set_idx(_get_ctx(), _c_ptr(self), _c_ptr(wrapped_val), key)
        elif isinstance(key, Tuple) and all(isinstance(i, int) for i in key):
            _dsc_tensor_set_idx(
                _get_ctx(),
                _c_ptr(self),
                _c_ptr(wrapped_val),
                *tuple(i for i in key),  # pyright: ignore[reportArgumentType]
            )
        elif isinstance(key, slice):
            _dsc_tensor_set_slice(
                _get_ctx(), _c_ptr(self), _c_ptr(wrapped_val), _c_slice(key)
            )
        elif (isinstance(key, Tuple) and all(isinstance(s, slice) for s in key)) or (
            isinstance(key, Tuple)
            and all(isinstance(s, int) or isinstance(s, slice) for s in key)
        ):
            _dsc_tensor_set_slice(
                _get_ctx(),
                _c_ptr(self),
                _c_ptr(wrapped_val),
                *tuple(_c_slice(s) for s in key),
            )
        else:
            raise RuntimeError(f'cannot set Tensor with index {key}')

    def __add__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return add(self, other)

    def __radd__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return add(other, self)

    def __sub__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return sub(self, other)

    def __rsub__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return sub(other, self)

    def __mul__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return mul(self, other)

    def __rmul__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return mul(other, self)

    def __truediv__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return true_div(self, other)

    def __rtruediv__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return true_div(other, self)

    def __pow__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return power(self, other)

    def __rpow__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return power(other, self)

    def __bytes__(self) -> bytes:
        byte_array = (ctypes.c_byte * self.ne * DTYPE_SIZE[self.dtype]).from_address(
            self._c_ptr.contents.data
        )
        return bytes(byte_array)

    def numpy(self) -> np.ndarray:
        # Note: this method could become the source of some nasty bugs. Here, we are creating a NumPy array
        # that is a view of some data managed by DSC. It can happen that the underlying DSC buffer is freed before
        # the NumPy array itself. For now, since we are using this basically just to verify that two arrays match, it's
        # not a problem but it's worth keeping an eye out for future bugs.
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
        return Tensor(_dsc_cast(_get_ctx(), self._c_ptr, dtype))

    def tobytes(self) -> bytes:
        return bytes(self)

    def reshape(self, *shape: Union[int, Tuple[int, ...], List[int]]) -> 'Tensor':
        return reshape(self, *shape)


def _create_tensor(dtype: Dtype, *dims: int) -> Tensor:
    n_dims = len(dims)
    if n_dims > _DSC_MAX_DIMS or n_dims < 1:
        raise RuntimeError(
            f'cannot create a Tensor with {n_dims} dimensions, max is {_DSC_MAX_DIMS}'
        )

    if n_dims == 1:
        return Tensor(_dsc_tensor_1d(_get_ctx(), dtype, dims[0]))
    elif n_dims == 2:
        return Tensor(_dsc_tensor_2d(_get_ctx(), dtype, dims[0], dims[1]))
    elif n_dims == 3:
        return Tensor(
            _dsc_tensor_3d(
                _get_ctx(),
                dtype,
                dims[0],
                dims[1],
                dims[2],
            )
        )
    else:
        return Tensor(
            _dsc_tensor_4d(
                _get_ctx(),
                dtype,
                dims[0],
                dims[1],
                dims[2],
                dims[3],
            )
        )


def from_numpy(x: np.ndarray) -> Tensor:
    if x.dtype not in NP_TO_DTYPE:
        raise RuntimeError(f'NumPy dtype {x.dtype} is not supported')

    out = _create_tensor(NP_TO_DTYPE[x.dtype], *x.shape)
    ctypes.memmove(_c_ptr(out).contents.data, x.ctypes.data, x.nbytes)
    return out


def reshape(x: Tensor, *shape: Union[int, Tuple[int, ...], List[int]]) -> Tensor:
    if (
        len(shape) == 1
        and isinstance(shape[0], (Tuple, List))
        and all(isinstance(s, int) for s in shape[0])
    ):
        shape_tuple = tuple(shape[0])
    elif all(isinstance(s, int) for s in shape):
        shape_tuple = shape
    else:
        raise RuntimeError(f'cannot reshape Tensor with shape {shape}')
    return Tensor(_dsc_reshape(_get_ctx(), _c_ptr(x), *shape_tuple))  # pyright: ignore[reportArgumentType]


def _tensor_op(
    xa: Tensor, xb: Tensor, out: Union[Tensor, None], op_name: str
) -> Tensor:
    if hasattr(sys.modules[__name__], op_name):
        op = getattr(sys.modules[__name__], op_name)
        return Tensor(op(_get_ctx(), _c_ptr(xa), _c_ptr(xb), _c_ptr_or_none(out)))
    else:
        raise RuntimeError(f'tensor operation "{op_name}" doesn\'t exist in module')


def _wrap_operands(
    xa: Union[ScalarType, TensorType], xb: Union[ScalarType, TensorType]
) -> Tuple[Tensor, Tensor]:
    def _dtype(x: Union[ScalarType, TensorType]) -> Dtype:
        if isinstance(x, Tensor):
            return x.dtype
        elif isinstance(x, np.ndarray):
            return NP_TO_DTYPE[x.dtype]
        elif isinstance(x, int) or isinstance(x, float):
            return Dtype.F32
        else:
            return Dtype.C32

    if (isinstance(xa, Tensor) and isinstance(xb, Tensor)) or (
        isinstance(xa, np.ndarray) and isinstance(xb, np.ndarray)
    ):
        return _wrap(xa), _wrap(xb)

    xa_dtype = _dtype(xa)
    xb_dtype = _dtype(xb)
    wrap_dtype = DTYPE_CONVERSION_TABLES[xa_dtype.value][xb_dtype.value]
    return _wrap(xa, wrap_dtype), _wrap(xb, wrap_dtype)


def add(
    xa: Union[ScalarType, TensorType],
    xb: Union[ScalarType, TensorType],
    out: Union[Tensor, None] = None,
) -> Tensor:
    xa, xb = _wrap_operands(xa, xb)
    return _tensor_op(xa, xb, out, op_name='_dsc_add')


def sub(
    xa: Union[ScalarType, TensorType],
    xb: Union[ScalarType, TensorType],
    out: Union[Tensor, None] = None,
) -> Tensor:
    xa, xb = _wrap_operands(xa, xb)
    return _tensor_op(xa, xb, out, op_name='_dsc_sub')


def mul(
    xa: Union[ScalarType, TensorType],
    xb: Union[ScalarType, TensorType],
    out: Union[Tensor, None] = None,
) -> Tensor:
    xa, xb = _wrap_operands(xa, xb)
    return _tensor_op(xa, xb, out, op_name='_dsc_mul')


def true_div(
    xa: Union[ScalarType, TensorType],
    xb: Union[ScalarType, TensorType],
    out: Union[Tensor, None] = None,
) -> Tensor:
    xa, xb = _wrap_operands(xa, xb)
    return _tensor_op(xa, xb, out, op_name='_dsc_div')


def power(
    xa: Union[ScalarType, TensorType],
    xb: Union[ScalarType, TensorType],
    out: Union[Tensor, None] = None,
) -> Tensor:
    xa, xb = _wrap_operands(xa, xb)
    return _tensor_op(xa, xb, out, op_name='_dsc_pow')


def cos(x: Tensor, out: Union[Tensor, None] = None) -> Tensor:
    return Tensor(_dsc_cos(_get_ctx(), _c_ptr(x), _c_ptr_or_none(out)))


def sin(x: Tensor, out: Union[Tensor, None] = None) -> Tensor:
    return Tensor(_dsc_sin(_get_ctx(), _c_ptr(x), _c_ptr_or_none(out)))


def sinc(x: Tensor, out: Union[Tensor, None] = None) -> Tensor:
    return Tensor(_dsc_sinc(_get_ctx(), _c_ptr(x), _c_ptr_or_none(out)))


def logn(x: Tensor, out: Union[Tensor, None] = None) -> Tensor:
    return Tensor(_dsc_logn(_get_ctx(), _c_ptr(x), _c_ptr_or_none(out)))


def log2(x: Tensor, out: Union[Tensor, None] = None) -> Tensor:
    return Tensor(_dsc_log2(_get_ctx(), _c_ptr(x), _c_ptr_or_none(out)))


def log10(x: Tensor, out: Union[Tensor, None] = None) -> Tensor:
    return Tensor(_dsc_log10(_get_ctx(), _c_ptr(x), _c_ptr_or_none(out)))


def exp(x: Tensor, out: Union[Tensor, None] = None) -> Tensor:
    return Tensor(_dsc_exp(_get_ctx(), _c_ptr(x), _c_ptr_or_none(out)))


def sqrt(x: Tensor, out: Union[Tensor, None] = None) -> Tensor:
    return Tensor(_dsc_sqrt(_get_ctx(), _c_ptr(x), _c_ptr_or_none(out)))


def absolute(x: Tensor, out: Union[Tensor, None] = None) -> Tensor:
    return Tensor(_dsc_abs(_get_ctx(), _c_ptr(x), _c_ptr_or_none(out)))


def angle(x: Tensor) -> Tensor:
    return Tensor(_dsc_angle(_get_ctx(), _c_ptr(x)))


def conj(x: Tensor) -> Tensor:
    return Tensor(_dsc_conj(_get_ctx(), _c_ptr(x)))


def real(x: Tensor) -> Tensor:
    return Tensor(
        _dsc_real(
            _get_ctx(),
            _c_ptr(x),
        )
    )


def imag(x: Tensor) -> Tensor:
    return Tensor(_dsc_imag(_get_ctx(), _c_ptr(x)))


def i0(x: Union[int, float, Tensor], dtype: Dtype = Dtype.F32) -> Tensor:
    x = _wrap(x, dtype)
    return Tensor(_dsc_i0(_get_ctx(), _c_ptr(x)))


def clip(
    x: Tensor,
    x_min: Union[float, None] = None,
    x_max: Union[float, None] = None,
    out: Union[Tensor, None] = None,
) -> Tensor:
    x_min = x_min if x_min is not None else float('-inf')
    x_max = x_max if x_max is not None else float('+inf')
    return Tensor(_dsc_clip(_get_ctx(), _c_ptr(x), _c_ptr_or_none(out), x_min, x_max))


def sum(
    x: Tensor, out: Union[Tensor, None] = None, axis: int = -1, keepdims: bool = True
) -> Tensor:
    return Tensor(_dsc_sum(_get_ctx(), _c_ptr(x), _c_ptr_or_none(out), axis, keepdims))


def mean(
    x: Tensor, out: Union[Tensor, None] = None, axis: int = -1, keepdims: bool = True
) -> Tensor:
    return Tensor(_dsc_mean(_get_ctx(), _c_ptr(x), _c_ptr_or_none(out), axis, keepdims))


def max(
    x: Tensor, out: Union[Tensor, None] = None, axis: int = -1, keepdims: bool = True
) -> Tensor:
    return Tensor(_dsc_max(_get_ctx(), _c_ptr(x), _c_ptr_or_none(out), axis, keepdims))


def min(
    x: Tensor, out: Union[Tensor, None] = None, axis: int = -1, keepdims: bool = True
) -> Tensor:
    return Tensor(_dsc_min(_get_ctx(), _c_ptr(x), _c_ptr_or_none(out), axis, keepdims))


def arange(n: int, dtype: Dtype = Dtype.F32) -> Tensor:
    return Tensor(_dsc_arange(_get_ctx(), n, dtype))


def randn(*shape: int, dtype: Dtype = Dtype.F32) -> Tensor:
    return Tensor(_dsc_randn(_get_ctx(), shape, dtype))


# In the xx_like methods if dtype is not specified it will be the same as x
def ones(
    shape: Union[int, Tuple[int, ...], List[int]], dtype: Dtype = Dtype.F32
) -> Tensor:
    return full(shape, fill_value=1, dtype=dtype)


def ones_like(x: TensorType, dtype: Union[Dtype, None] = None) -> Tensor:
    return full_like(x, 1, dtype)


def zeros(
    shape: Union[int, Tuple[int, ...], List[int]], dtype: Dtype = Dtype.F32
) -> Tensor:
    return full(shape, fill_value=0, dtype=dtype)


def zeros_like(x: TensorType, dtype: Union[Dtype, None] = None) -> Tensor:
    return full_like(x, fill_value=0, dtype=dtype)


def full(
    shape: Union[int, Tuple[int, ...], List[int]],
    fill_value: ScalarType,
    dtype: Dtype = Dtype.F32,
) -> Tensor:
    shape = (shape,) if isinstance(shape, int) else tuple(i for i in shape)
    out = _create_tensor(dtype, *shape)
    out[:] = fill_value
    return out


def full_like(
    x: TensorType, fill_value: ScalarType, dtype: Union[Dtype, None] = None
) -> Tensor:
    shape = x.shape
    dtype = (
        dtype
        if dtype is not None
        else (x.dtype if isinstance(x, Tensor) else NP_TO_DTYPE[x.dtype])
    )
    return full(shape, fill_value=fill_value, dtype=dtype)


def empty(
    shape: Union[int, Tuple[int, ...], List[int]], dtype: Dtype = Dtype.F32
) -> Tensor:
    shape = (shape,) if isinstance(shape, int) else tuple(i for i in shape)
    return _create_tensor(dtype, *shape)


def empty_like(x: TensorType, dtype: Union[Dtype, None] = None) -> Tensor:
    shape = x.shape
    dtype = (
        dtype
        if dtype is not None
        else (x.dtype if isinstance(x, Tensor) else NP_TO_DTYPE[x.dtype])
    )
    return empty(shape, dtype=dtype)


def plan_fft(n: int, dtype: Dtype = Dtype.F64):
    """
    Create the plan for a one-dimensional FFT/IFFT of size N using dtype for the twiddle factors.
    If this function is not executed before calling either `dsc.fft` or `dsc.ifft` then it will be
    called automatically before doing the first transform causing a slowdown.
    """
    return _dsc_plan_fft(_get_ctx(), n, dtype)


def fft(
    x: Tensor, out: Union[Tensor, None] = None, n: int = -1, axis: int = -1
) -> Tensor:
    return Tensor(_dsc_fft(_get_ctx(), _c_ptr(x), _c_ptr_or_none(out), n=n, axis=axis))


def ifft(
    x: Tensor, out: Union[Tensor, None] = None, n: int = -1, axis: int = -1
) -> Tensor:
    return Tensor(_dsc_ifft(_get_ctx(), _c_ptr(x), _c_ptr_or_none(out), n=n, axis=axis))


def rfft(
    x: Tensor, out: Union[Tensor, None] = None, n: int = -1, axis: int = -1
) -> Tensor:
    return Tensor(_dsc_rfft(_get_ctx(), _c_ptr(x), _c_ptr_or_none(out), n=n, axis=axis))


def irfft(
    x: Tensor, out: Union[Tensor, None] = None, n: int = -1, axis: int = -1
) -> Tensor:
    return Tensor(
        _dsc_irfft(_get_ctx(), _c_ptr(x), _c_ptr_or_none(out), n=n, axis=axis)
    )


def fftfreq(n: int, d: float = 1.0, dtype: Dtype = Dtype.F32) -> Tensor:
    return Tensor(_dsc_fftfreq(_get_ctx(), n, d, dtype))


def rfftfreq(n: int, d: float = 1.0, dtype: Dtype = Dtype.F32) -> Tensor:
    return Tensor(_dsc_rfftfreq(_get_ctx(), n, d, dtype))
