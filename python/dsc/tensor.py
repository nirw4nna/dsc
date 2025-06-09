# Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
# All rights reserved.
#
# This code is licensed under the terms of the 3-clause BSD license
# (https://opensource.org/license/bsd-3-clause).

from ._bindings import (
    _DscTensor_p,
    _DscDataBuffer_p,
    _OptionalTensor,
    _DSC_MAX_DIMS,
    _DSC_VALUE_NONE,
    _DscSlice,
    _DscComparison,
    _dsc_cast,
    _dsc_reshape,
    _dsc_concat,
    _dsc_compare,
    _dsc_where,
    _dsc_masked_fill,
    _dsc_outer,
    _dsc_multinomial,
    _dsc_transpose,
    _dsc_tril,
    _dsc_tensor_free,
    _dsc_sum,
    _dsc_max,
    _dsc_min,
    _dsc_tensor_get_idx,
    _dsc_tensor_get_slice,
    _dsc_tensor_get_tensor,
    _dsc_tensor_set_idx,
    _dsc_tensor_set_slice,
    _dsc_tensor_set_buffer,
    _dsc_view,
    _dsc_copy,
    _dsc_to,
    _dsc_wrap_bool,
    _dsc_wrap_i32,
    _dsc_wrap_f32,
    _dsc_wrap_f64,
    _dsc_add,
    _dsc_sub,
    _dsc_mul,
    _dsc_div,
    _dsc_pow,
    _dsc_matmul,
    _dsc_arange,
    _dsc_repeat,
    _dsc_randn,
    _dsc_cos,
    _dsc_sin,
    _dsc_tanh,
    _dsc_exp,
    _dsc_sqrt,
    _dsc_topk,
    _dsc_tensor_1d,
    _dsc_tensor_2d,
    _dsc_tensor_3d,
    _dsc_tensor_4d,
)
from .dtype import (
    Dtype,
    NP_TO_DTYPE,
    DTYPE_TO_NP,
    DTYPE_CONVERSION_TABLES,
    DTYPE_SIZE,
    DTYPE_TO_CTYPE,
    ScalarType
)
from .device import Device, DeviceType, _get_device
import numpy as np
from .context import _get_ctx
import ctypes
import sys
from typing import Union, Tuple, List, Optional, Any, Iterable


TensorType = Union['Tensor', np.ndarray]


def _unwrap(x: 'Tensor') -> Union[ScalarType, 'Tensor']:
    # If x is not wrapping a single value return it
    if x.n_dim != 1 or len(x) != 1:
        return x
    # If x is a scalar on the GPU move it to to CPU first
    x = x.to('cpu')
    return ctypes.cast(x.data, DTYPE_TO_CTYPE[x.dtype]).contents.value


def _c_slice(x: Union[slice, int]) -> _DscSlice:
    def _sanitize(i: Optional[int]) -> int:
        if i is None:
            return _DSC_VALUE_NONE
        return i

    if isinstance(x, slice):
        return _DscSlice(_sanitize(x.start), _sanitize(x.stop), _sanitize(x.step))
    else:
        # This is needed for when indexes and slices are mixed: suppose x[-1, ::-2] since we don't have a method that
        # takes both indexes and slices this will generate a new slice (-1) where start=dim stop=dim step=dim. All the
        # handling will be done internally by the library.
        return _DscSlice(x, x, x)


def _wrap(
    x: Union[ScalarType, TensorType], dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> 'Tensor':
    device = device if device is not None else Device.DEFAULT
    if isinstance(x, np.ndarray):
        # The conversion from (and to) NumPy is not free, so it's better to do that once and then work with DSC tensors.
        # This is here just for convenience not because it's a best practice.
        return from_numpy(x)
    elif isinstance(x, Tensor):
        return x

    if isinstance(x, bool):
        x_dtype = Dtype.BOOL
    elif isinstance(x, int):
        x_dtype = Dtype.I32
    elif isinstance(x, float):
        x_dtype = Dtype.F32

    if dtype is not None:
        out_dtype = dtype
    else:
        out_dtype = x_dtype

    if out_dtype == Dtype.BOOL:
        return Tensor(_dsc_wrap_bool(_get_ctx(), bool(x), device))
    elif out_dtype == Dtype.I32:
        return Tensor(_dsc_wrap_i32(_get_ctx(), int(x), device))
    elif out_dtype == Dtype.F64:
        return Tensor(_dsc_wrap_f64(_get_ctx(), float(x), device))
    else:
        # Default to F32
        return Tensor(_dsc_wrap_f32(_get_ctx(), float(x), device))


def _pointers_are_equals(xa: _DscTensor_p, xb: _DscTensor_p) -> bool:
    return (
        ctypes.cast(xa, ctypes.c_void_p).value == ctypes.cast(xb, ctypes.c_void_p).value
    )


class Tensor:
    def __init__(self, c_ptr: _DscTensor_p, view: bool = False):  # pyright: ignore[reportInvalidTypeForm]
        c_ptr = c_ptr if not view else _dsc_view(_get_ctx(), c_ptr)
        self._dtype = Dtype(c_ptr.contents.dtype)
        self._device = Device(c_ptr.contents.device)
        self._shape = c_ptr.contents.shape
        self._n_dim = c_ptr.contents.n_dim
        self._ne = c_ptr.contents.ne
        self._buf = c_ptr.contents.buf
        self._c_ptr = c_ptr

    def __del__(self):
        _dsc_tensor_free(_get_ctx(), self.c_ptr)

    @property
    def dtype(self) -> Dtype:
        return self._dtype

    @property
    def device(self) -> Device:
        return self._device

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._shape[_DSC_MAX_DIMS - self.n_dim :])

    @property
    def n_dim(self) -> int:
        return self._n_dim

    @property
    def ne(self) -> int:
        return self._ne

    @property
    def data(self) -> int:
        if self.device is not Device.CPU:
            raise RuntimeError('can\'t access _data field for a non-CPU tensor')
        return self._buf.contents.data

    @property
    def c_ptr(self) -> _DscTensor_p:
        return self._c_ptr

    @property
    def buf_ptr(self) -> _DscDataBuffer_p:
        return self._buf

    def size(self, axis: int) -> int:
        return self.shape[axis]

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
            'Tensor',
        ],
    ) -> Union[ScalarType, 'Tensor']:
        if isinstance(item, int):
            return _unwrap(Tensor(_dsc_tensor_get_idx(_get_ctx(), self.c_ptr, item)))
        elif isinstance(item, Tuple) and all(isinstance(i, int) for i in item):
            return _unwrap(
                Tensor(
                    _dsc_tensor_get_idx(
                        _get_ctx(),
                        self.c_ptr,
                        *tuple(i for i in item),  # pyright: ignore[reportArgumentType]
                    )
                )
            )
        elif isinstance(item, slice):
            return Tensor(
                _dsc_tensor_get_slice(_get_ctx(), self.c_ptr, _c_slice(item))
            )
        elif (isinstance(item, Tuple) and all(isinstance(s, slice) for s in item)) or (
            isinstance(item, Tuple)
            and all(isinstance(i, int) or isinstance(i, slice) for i in item)
        ):
            return Tensor(
                _dsc_tensor_get_slice(
                    _get_ctx(), self.c_ptr, *tuple(_c_slice(s) for s in item)
                )
            )
        elif isinstance(item, Tensor):
            return Tensor(
                _dsc_tensor_get_tensor(
                    _get_ctx(), self.c_ptr, item.c_ptr
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
        wrapped_val = _wrap(value, self.dtype, self.device)
        if isinstance(key, int):
            _dsc_tensor_set_idx(_get_ctx(), self.c_ptr, wrapped_val.c_ptr, key)
        elif isinstance(key, Tuple) and all(isinstance(i, int) for i in key):
            _dsc_tensor_set_idx(
                _get_ctx(),
                self.c_ptr,
                wrapped_val.c_ptr,
                *tuple(i for i in key),  # pyright: ignore[reportArgumentType]
            )
        elif isinstance(key, slice):
            _dsc_tensor_set_slice(
                _get_ctx(), self.c_ptr, wrapped_val.c_ptr, _c_slice(key)
            )
        elif (isinstance(key, Tuple) and all(isinstance(s, slice) for s in key)) or (
            isinstance(key, Tuple)
            and all(isinstance(s, int) or isinstance(s, slice) for s in key)
        ):
            _dsc_tensor_set_slice(
                _get_ctx(),
                self.c_ptr,
                wrapped_val.c_ptr,
                *tuple(_c_slice(s) for s in key),
            )
        else:
            raise RuntimeError(f'cannot set Tensor with index {key}')

    def __add__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return add(self, other)

    def __radd__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return add(other, self)

    def __iadd__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return add(self, other, self)

    def __sub__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return sub(self, other)

    def __rsub__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return sub(other, self)

    def __isub__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return sub(self, other, self)

    def __mul__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return mul(self, other)

    def __rmul__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return mul(other, self)

    def __imul__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return mul(self, other, self)

    def __truediv__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return true_div(self, other)

    def __rtruediv__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return true_div(other, self)

    def __itruediv__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return true_div(self, other, self)

    def __pow__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return power(self, other)

    def __rpow__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return power(other, self)

    def __ipow__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return power(self, other, self)

    def __matmul__(self, other: TensorType) -> 'Tensor':
        return matmul(self, other)

    def __rmatmul__(self, other: TensorType) -> 'Tensor':
        return matmul(other, self)

    def __eq__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return equal(self, other)

    def __ne__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return not_equal(self, other)
    
    def __lt__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return less(self, other)

    def __le__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return less_equal(self, other)

    def __gt__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return greater(self, other)

    def __ge__(self, other: Union[ScalarType, TensorType]) -> 'Tensor':
        return greater_equal(self, other)

    def __neg__(self) -> 'Tensor':
        return -1 * self

    def __bytes__(self) -> bytes:
        byte_array = (ctypes.c_byte * self.ne * DTYPE_SIZE[self.dtype]).from_address(self.data)
        return bytes(byte_array)

    def numpy(self) -> np.ndarray:
        self_cpu = self.to('cpu')

        typed_data = ctypes.cast(self_cpu.data, DTYPE_TO_CTYPE[self.dtype])

        # Create a copy of the underlying data buffer
        np_array = np.ctypeslib.as_array(typed_data, shape=self.shape).copy()

        return np_array.reshape(self.shape)

    def load(self, x: TensorType):
        if isinstance(x, np.ndarray):
            assert x.shape == self.shape
            assert NP_TO_DTYPE[x.dtype] == self.dtype

            data_ptr = ctypes.cast(x.ctypes.data, ctypes.c_void_p)
            _dsc_copy(_get_ctx(), self.c_ptr, data_ptr, x.size * DTYPE_SIZE[NP_TO_DTYPE[x.dtype]], self.device)
        elif isinstance(x, Tensor):
            _dsc_tensor_set_buffer(_get_ctx(), self.c_ptr, x.buf_ptr)
        else:
            raise RuntimeError(f'invalid argument for load {type(x)}')

    def cast(self, dtype: Dtype) -> 'Tensor':
        if self.dtype == dtype:
            return self
        out_ptr = _dsc_cast(_get_ctx(), self.c_ptr, dtype)
        return Tensor(out_ptr, _pointers_are_equals(self.c_ptr, out_ptr))

    def to(self, device: DeviceType) -> 'Tensor':
        device = _get_device(device)
        if self.device == device:
            return self
        return Tensor(_dsc_to(_get_ctx(), self.c_ptr, device))

    def tobytes(self) -> bytes:
        return bytes(self)

    def reshape(self, *shape: Union[int, Tuple[int, ...], List[int]]) -> 'Tensor':
        return reshape(self, *shape )
    
    def transpose(self, axes: Optional[Union[Tuple[int, ...], List[int]]] = None) -> 'Tensor':
        return transpose(self, axes)

    def masked_fill(self, mask: TensorType, value: float) -> 'Tensor':
        mask = _wrap(mask)
        _dsc_masked_fill(_get_ctx(), self.c_ptr, mask.c_ptr, value)
        return self

    def split(self, ne: int, axis: int = -1) -> Tuple['Tensor', ...]:
        return split(self, ne, axis)

    def mean(self, axis: int = -1, keepdims: bool = True) -> 'Tensor':
        return mean(self, None, axis=axis, keepdims=keepdims)

    def var(self, axis: int = -1, keepdims: bool = True) -> 'Tensor':
        return var(self, None, axis=axis, keepdims=keepdims)


def _create_tensor(dtype: Dtype, dims: Tuple[int, ...], device: Device, data: ctypes.c_void_p = None) -> Tensor:
    n_dims = len(dims)
    if n_dims > _DSC_MAX_DIMS or n_dims < 1:
        raise RuntimeError(
            f'cannot create a Tensor with {n_dims} dimensions, max is {_DSC_MAX_DIMS}'
        )

    if n_dims == 1:
        return Tensor(_dsc_tensor_1d(_get_ctx(), dtype, dims[0], device, data, Device.CPU))
    elif n_dims == 2:
        return Tensor(_dsc_tensor_2d(_get_ctx(), dtype, dims[0], dims[1], device, data, Device.CPU))
    elif n_dims == 3:
        return Tensor(
            _dsc_tensor_3d(
                _get_ctx(),
                dtype,
                dims[0],
                dims[1],
                dims[2],
                device,
                data,
                Device.CPU
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
                device,
                data,
                Device.CPU
            )
        )


def from_numpy(x: np.ndarray, device: DeviceType = Device.DEFAULT) -> Tensor:
    if x.dtype not in NP_TO_DTYPE:
        raise RuntimeError(f'NumPy dtype {x.dtype} is not supported')

    x_ptr = ctypes.cast(x.ctypes.data, ctypes.c_void_p)
    out = _create_tensor(NP_TO_DTYPE[x.dtype], x.shape, _get_device(device), x_ptr)
    return out


def tensor(x: Iterable, dtype: Dtype, device: DeviceType = Device.DEFAULT) -> Tensor:
    x_ = np.array(x, dtype=DTYPE_TO_NP[dtype])
    return from_numpy(x_, device)


def from_buffer(shape: Tuple[int, ...], dtype: Dtype, data: ctypes.c_void_p, device: DeviceType = Device.DEFAULT) -> Tensor:
    return _create_tensor(dtype, shape, _get_device(device), data)


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
        raise RuntimeError(f'cannot reshape tensor with shape {shape}')
    return Tensor(_dsc_reshape(_get_ctx(), x.c_ptr, *shape_tuple))  # pyright: ignore[reportArgumentType]


def concat(
    tensors: Union[Tuple[Tensor, ...], List[Tensor]], axis: Optional[int] = 0
) -> Tensor:
    if isinstance(tensors, (Tuple, List)) and all(
        isinstance(t, Tensor) for t in tensors
    ):
        tensors_tuple = tuple(t.c_ptr for t in tensors)
        axis = axis if axis is not None else _DSC_VALUE_NONE
        return Tensor(_dsc_concat(_get_ctx(), axis, *tensors_tuple))
    else:
        raise RuntimeError(f'cannot concatenate tensors {tensors}')


def split(x: Tensor, ne: int, axis: int = -1) -> Tuple[Tensor, ...]:
    n = x.size(axis)
    if n % ne != 0:
        raise RuntimeError(f'cannot split {x.shape} along {axis} ({n} is not a multiple of {ne})')
    slices = []
    for idx in range(n // ne):
        s = [slice(None, None, 1)] * x.n_dim
        s[axis] = slice(idx * ne, (idx + 1) * ne, 1)
        slices.append(tuple(s))

    return tuple(x[s] for s in slices)


def transpose(
    x: Tensor, axes: Optional[Union[Tuple[int, ...], List[int]]] = None
) -> Tensor:
    if (isinstance(axes, (Tuple, List)) and all(isinstance(a, int) for a in axes)) or (
        axes is None
    ):
        axes_tuple = tuple(axes) if axes is not None else tuple()
        return Tensor(_dsc_transpose(_get_ctx(), x._c_ptr, *axes_tuple))
    else:
        raise RuntimeError(f'cannot transpose axes {axes}')


def tril(x: TensorType, diagonal: int = 0, out: Optional[Tensor] = None) -> Tensor:
    x = _wrap(x)
    return Tensor(_dsc_tril(_get_ctx(), x.c_ptr, diagonal, _c_ptr_or_none(out)), _has_out(out))


def _c_ptr_or_none(x: Optional['Tensor']) -> _OptionalTensor:
    return x.c_ptr if x else None


def _has_out(out: Optional[Tensor]) -> bool:
    return True if out is not None else False


def _get_op(op_name: str) -> Optional[Any]:
    if hasattr(sys.modules[__name__], op_name):
        return getattr(sys.modules[__name__], op_name)
    else:
        raise RuntimeError(f'tensor operation "{op_name}" doesn\'t exist in module')


def _binary_op(
    xa: Union[ScalarType, TensorType],
    xb: Union[ScalarType, TensorType],
    out: Optional[Tensor],
    op_name: str,
    comp: Optional[_DscComparison] = None
) -> Tensor:
    xa, xb = _wrap_operands(xa, xb)
    op = _get_op(op_name)
    if comp is None:
        return Tensor(
            op(_get_ctx(), xa.c_ptr, xb.c_ptr, _c_ptr_or_none(out)), _has_out(out)
        )
    else:
        return Tensor(
            op(_get_ctx(), xa.c_ptr, xb.c_ptr, comp, _c_ptr_or_none(out)), _has_out(out)
        )


def _unary_op(
    x: Tensor,
    out: Optional[Tensor],
    op_name: str
) -> Tensor:
    op = _get_op(op_name)
    return Tensor(
        op(_get_ctx(), x.c_ptr, _c_ptr_or_none(out)), _has_out(out)
    )


def _reduction_op(
        x: Tensor,
        out: Optional[Tensor],
        axis: int,
        keepdims: bool,
        op_name: str
) -> Tensor:
    op = _get_op(op_name)
    return Tensor(
        op(_get_ctx(), x.c_ptr, _c_ptr_or_none(out), axis, keepdims), _has_out(out)
    )


def _wrap_operands(
    xa: Union[ScalarType, TensorType], xb: Union[ScalarType, TensorType]
) -> Tuple[Tensor, Tensor]:
    def _dtype(x: Union[ScalarType, TensorType]) -> Dtype:
        if isinstance(x, Tensor):
            return x.dtype
        elif isinstance(x, np.ndarray):
            return NP_TO_DTYPE[x.dtype]
        elif isinstance(x, bool):
            return Dtype.BOOL
        elif isinstance(x, int):
            return Dtype.I32
        else:
            return Dtype.F32

    def _device(x: Union[ScalarType, TensorType]) -> Optional[Device]:
        if isinstance(x, Tensor):
            return x.device
        elif isinstance(x, np.ndarray):
            return Device.CPU
        else:
            return None
    
    if (isinstance(xa, Tensor) and isinstance(xb, Tensor)) or (
        isinstance(xa, np.ndarray) and isinstance(xb, np.ndarray)
    ):
        return _wrap(xa), _wrap(xb)

    xa_device = _device(xa)
    xb_device = _device(xb)
    target_device = xa_device if xa_device is not None else xb_device

    xa_dtype = _dtype(xa)
    xb_dtype = _dtype(xb)
    wrap_dtype = DTYPE_CONVERSION_TABLES[xa_dtype.value][xb_dtype.value]
    return _wrap(xa, wrap_dtype, target_device), _wrap(xb, wrap_dtype, target_device)


def add(
    xa: Union[ScalarType, TensorType],
    xb: Union[ScalarType, TensorType],
    out: Optional[Tensor] = None,
) -> Tensor:
    return _binary_op(xa, xb, out, op_name='_dsc_add')


def sub(
    xa: Union[ScalarType, TensorType],
    xb: Union[ScalarType, TensorType],
    out: Optional[Tensor] = None,
) -> Tensor:
    return _binary_op(xa, xb, out, op_name='_dsc_sub')


def mul(
    xa: Union[ScalarType, TensorType],
    xb: Union[ScalarType, TensorType],
    out: Optional[Tensor] = None,
) -> Tensor:
    return _binary_op(xa, xb, out, op_name='_dsc_mul')


def true_div(
    xa: Union[ScalarType, TensorType],
    xb: Union[ScalarType, TensorType],
    out: Optional[Tensor] = None,
) -> Tensor:
    return _binary_op(xa, xb, out, op_name='_dsc_div')


def power(
    xa: Union[ScalarType, TensorType],
    xb: Union[ScalarType, TensorType],
    out: Optional[Tensor] = None,
) -> Tensor:
    return _binary_op(xa, xb, out, op_name='_dsc_pow')


def matmul(
    xa: TensorType,
    xb: TensorType,
    trans_b: bool = False,
    out: Optional[Tensor] = None,
) -> Tensor:
    xa, xb = _wrap_operands(xa, xb)
    return Tensor(_dsc_matmul(_get_ctx(), xa.c_ptr, xb.c_ptr, trans_b, _c_ptr_or_none(out)), _has_out(out))


def outer(
    xa: Union[ScalarType, TensorType],
    xb: Union[ScalarType, TensorType],
    out: Optional[Tensor] = None
) -> Tensor:
    return _binary_op(xa, xb, out, op_name='_dsc_outer')


def where(
    condition: TensorType,
    input: Union[ScalarType, TensorType],
    other: Union[ScalarType, TensorType],
    out: Optional[Tensor] = None
) -> Tensor:
    condition_ = _wrap(condition)
    input_, other_ = _wrap_operands(input, other)
    return Tensor(_dsc_where(_get_ctx(), condition_.c_ptr, input_.c_ptr, other_.c_ptr, _c_ptr_or_none(out)), _has_out(out))


def equal(
    xa: Union[ScalarType, TensorType],
    xb: Union[ScalarType, TensorType],
    out: Optional[Tensor] = None,
) -> Tensor:
    return _binary_op(xa, xb, out, op_name='_dsc_compare', comp=_DscComparison.EQ)


def not_equal(
    xa: Union[ScalarType, TensorType],
    xb: Union[ScalarType, TensorType],
    out: Optional[Tensor] = None,
) -> Tensor:
    return _binary_op(xa, xb, out, op_name='_dsc_compare', comp=_DscComparison.NE)


def less(
    xa: Union[ScalarType, TensorType],
    xb: Union[ScalarType, TensorType],
    out: Optional[Tensor] = None,
) -> Tensor:
    return _binary_op(xa, xb, out, op_name='_dsc_compare', comp=_DscComparison.LT)


def less_equal(
    xa: Union[ScalarType, TensorType],
    xb: Union[ScalarType, TensorType],
    out: Optional[Tensor] = None,
) -> Tensor:
    return _binary_op(xa, xb, out, op_name='_dsc_compare', comp=_DscComparison.LE)


def greater(
    xa: Union[ScalarType, TensorType],
    xb: Union[ScalarType, TensorType],
    out: Optional[Tensor] = None,
) -> Tensor:
    return _binary_op(xa, xb, out, op_name='_dsc_compare', comp=_DscComparison.GT)


def greater_equal(
    xa: Union[ScalarType, TensorType],
    xb: Union[ScalarType, TensorType],
    out: Optional[Tensor] = None,
) -> Tensor:
    return _binary_op(xa, xb, out, op_name='_dsc_compare', comp=_DscComparison.GE)


def cos(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    return _unary_op(x, out, op_name='_dsc_cos')


def sin(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    return _unary_op(x, out, op_name='_dsc_sin')


def tanh(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    return _unary_op(x, out, op_name='_dsc_tanh')


def exp(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    return _unary_op(x, out, op_name='_dsc_exp')


def sqrt(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    return _unary_op(x, out, op_name='_dsc_sqrt')


def rsqrt(x: Tensor, out: Optional[Tensor] = None) -> Tensor:
    out = sqrt(x, out)
    return true_div(1, out, out)


def sum(
    x: Tensor, out: Optional[Tensor] = None, axis: int = -1, keepdims: bool = True
) -> Tensor:
    return _reduction_op(x, out, axis, keepdims, op_name='_dsc_sum')


def mean(
    x: Tensor, out: Optional[Tensor] = None, axis: int = -1, keepdims: bool = True
) -> Tensor:
    out = sum(x, out, axis, keepdims)
    out *= (1. / x.size(axis))
    return out


def var(
    x: Tensor, out: Optional[Tensor] = None, axis: int = -1, keepdims: bool = True
) -> Tensor:
    return mean((x - x.mean(axis, True)) ** 2, out, axis, keepdims)


def max(
    x: Tensor, out: Optional[Tensor] = None, axis: int = -1, keepdims: bool = True
) -> Tensor:
    return _reduction_op(x, out, axis, keepdims, op_name='_dsc_max')


def min(
    x: Tensor, out: Optional[Tensor] = None, axis: int = -1, keepdims: bool = True
) -> Tensor:
    return _reduction_op(x, out, axis, keepdims, op_name='_dsc_min')


def arange(
    stop: Union[int, float],
    start: Union[int, float] = 0,
    step: Union[int, float] = 1,
    dtype: Dtype = Dtype.I32,
    device: DeviceType = Device.DEFAULT
) -> Tensor:
    return Tensor(_dsc_arange(_get_ctx(), stop, start, step, dtype, _get_device(device)))


def repeat(x: Tensor, repeats: int, axis: int = -1) -> Tensor:
    return Tensor(_dsc_repeat(_get_ctx(), x.c_ptr, repeats, axis))


def randn(
    *shape: int, dtype: Dtype = Dtype.F32, device: DeviceType = Device.DEFAULT
) -> Tensor:
    # TODO: (6)
    return Tensor(_dsc_randn(_get_ctx(), shape, dtype, _get_device(device)))


def topk(x: Tensor, k: int, axis: int = -1, largest: bool = True) -> Tuple[Tensor, Tensor]:
    res = _dsc_topk(_get_ctx(), x.c_ptr, k, axis, largest)
    return Tensor(res.first), Tensor(res.second)


def multinomial(x: Tensor, num_samples: int) -> Tensor:
    return Tensor(_dsc_multinomial(_get_ctx(), x.c_ptr, num_samples))


# In the xx_like methods if dtype is not specified it will be the same as x
def ones(
    shape: Union[int, Tuple[int, ...], List[int]], dtype: Dtype = Dtype.F32, device: DeviceType = Device.DEFAULT
) -> Tensor:
    return full(shape, 1, dtype, device)


def ones_like(x: TensorType, dtype: Optional[Dtype] = None, device: DeviceType = Device.DEFAULT) -> Tensor:
    return full_like(x, 1, dtype, device)


def zeros(
    shape: Union[int, Tuple[int, ...], List[int]], dtype: Dtype = Dtype.F32, device: DeviceType = Device.DEFAULT
) -> Tensor:
    return full(shape, 0, dtype, device)


def zeros_like(x: TensorType, dtype: Optional[Dtype] = None, device: DeviceType = Device.DEFAULT) -> Tensor:
    return full_like(x, 0, dtype, device)


def full(
    shape: Union[int, Tuple[int, ...], List[int]],
    fill_value: ScalarType,
    dtype: Dtype = Dtype.F32,
    device: DeviceType = Device.DEFAULT
) -> Tensor:
    shape = (shape,) if isinstance(shape, int) else tuple(i for i in shape)
    out = _create_tensor(dtype, shape, _get_device(device))
    out[:] = fill_value
    return out


def full_like(
    x: TensorType, fill_value: ScalarType, dtype: Optional[Dtype] = None, device: DeviceType = Device.DEFAULT
) -> Tensor:
    shape = x.shape
    dtype = (
        dtype
        if dtype is not None
        else (x.dtype if isinstance(x, Tensor) else NP_TO_DTYPE[x.dtype])
    )
    return full(shape, fill_value, dtype, device)


def empty(
    shape: Union[int, Tuple[int, ...], List[int]], dtype: Dtype = Dtype.F32, device: DeviceType = Device.DEFAULT
) -> Tensor:
    shape = (shape,) if isinstance(shape, int) else tuple(i for i in shape)
    return _create_tensor(dtype, shape, _get_device(device))


def empty_like(x: TensorType, dtype: Optional[Dtype] = None, device: DeviceType = Device.DEFAULT) -> Tensor:
    shape = x.shape
    dtype = (
        dtype
        if dtype is not None
        else (x.dtype if isinstance(x, Tensor) else NP_TO_DTYPE[x.dtype])
    )
    return empty(shape, dtype, device)
