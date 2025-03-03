# Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
# All rights reserved.
#
# This code is licensed under the terms of the 3-clause BSD license
# (https://opensource.org/license/bsd-3-clause).

import os
import ctypes
from ctypes import (
    c_bool,
    c_char_p,
    c_int,
    c_int32,
    c_int8,
    c_uint8,
    c_size_t,
    c_float,
    c_double,
    c_void_p,
    Structure,
    POINTER,
)
from typing import Union, Tuple
from enum import Enum
from .dtype import Dtype
from .device import Device


_DSC_MAX_DIMS = 4
_DSC_VALUE_NONE = 2**31 - 1

_DscCtx = c_void_p

# Todo: make this more flexible
_lib_file = f'{os.path.dirname(__file__)}/libdsc.so'
if not os.path.exists(_lib_file):
    raise RuntimeError(f'Error loading DSC shared object "{_lib_file}"')

_lib = ctypes.CDLL(_lib_file)


class _DscDataBuffer(Structure):
    _fields_ = [
        ('data', c_void_p),
        ('size', c_size_t),
        ('refs', c_int),
    ]


class _DscTensor(Structure):
    _fields_ = [
        ('shape', c_int * _DSC_MAX_DIMS),
        ('stride', c_int * _DSC_MAX_DIMS),
        ('buf', POINTER(_DscDataBuffer)),
        ('ne', c_int),
        ('n_dim', c_int),
        ('dtype', c_uint8),
        ('device', c_int8),
    ]


class _DscComparison(Enum):
    EQ = 0
    NE = 1
    LT = 2
    LE = 3
    GT = 4
    GE = 5


_DscTensor_p_ = POINTER(_DscTensor)
# For some reason this format works fine with Pyright while just doing _DscTensor_p = POINTER(_DscTensor) doesn't
_DscTensor_p = _DscTensor_p_

_OptionalTensor = Union[_DscTensor_p, None]


class _DscSlice(Structure):
    _fields_ = [('start', c_int), ('stop', c_int), ('step', c_int)]


# extern dsc_ctx *dsc_ctx_init(usize main_mem, usize scratch_mem);
def _dsc_ctx_init(mem_size: int) -> _DscCtx:
    return _lib.dsc_ctx_init(c_size_t(mem_size))


_lib.dsc_ctx_init.argtypes = [c_size_t]
_lib.dsc_ctx_init.restype = _DscCtx


# extern void dsc_ctx_free(dsc_ctx *ctx);
def _dsc_ctx_free(ctx: _DscCtx):
    _lib.dsc_ctx_free(ctx)


_lib.dsc_ctx_free.argtypes = [_DscCtx]
_lib.dsc_ctx_free.restype = None


# extern void dsc_tensor_free(dsc_ctx *ctx, dsc_tensor *x);
def _dsc_tensor_free(ctx: _DscCtx, x: _DscTensor_p):
    _lib.dsc_tensor_free(ctx, x)


_lib.dsc_tensor_free.argtypes = [_DscCtx, _DscTensor_p]
_lib.dsc_tensor_free.restype = None


# extern usize dsc_used_mem(dsc_ctx *ctx);
def _dsc_used_mem(ctx: _DscCtx) -> int:
    return _lib.dsc_used_mem(ctx)


_lib.dsc_used_mem.argtypes = [_DscCtx]
_lib.dsc_used_mem.restype = c_size_t


# extern void dsc_print_mem_usage(dsc_ctx *ctx);
def _dsc_print_mem_usage(ctx: _DscCtx):
    _lib.dsc_print_mem_usage(ctx)


_lib.dsc_print_mem_usage.argtypes = [_DscCtx]
_lib.dsc_print_mem_usage.restype = None


# extern void dsc_traces_record(dsc_ctx *ctx, bool record);
def _dsc_traces_record(ctx: _DscCtx, record: bool):
    _lib.dsc_traces_record(ctx, c_bool(record))


_lib.dsc_traces_record.argtypes = [_DscCtx, c_bool]
_lib.dsc_traces_record.restype = None


# extern void dsc_dump_traces(dsc_ctx *ctx, const char *filename);
def _dsc_dump_traces(ctx: _DscCtx, filename: str):
    _lib.dsc_dump_traces(ctx, c_char_p(filename.encode('utf-8')))


_lib.dsc_dump_traces.argtypes = [_DscCtx, c_char_p]
_lib.dsc_dump_traces.restype = None


# extern void dsc_clear_traces(dsc_ctx *);
def _dsc_clear_traces(ctx: _DscCtx):
    _lib.dsc_clear_traces(ctx)


_lib.dsc_clear_traces.argtypes = [_DscCtx]
_lib.dsc_clear_traces.restype = None


# extern dsc_tensor *dsc_view(dsc_ctx *ctx,
#                             const dsc_tensor *x);
def _dsc_view(ctx: _DscCtx, x: _DscTensor_p) -> _DscTensor_p:
    return _lib.dsc_view(ctx, x)


_lib.dsc_view.argtypes = [_DscCtx, _DscTensor_p]
_lib.dsc_view.restype = _DscTensor_p


# extern dsc_tensor *dsc_new_tensor(dsc_ctx *ctx,
#                                   int n_dim,
#                                   const int *shape,
#                                   dsc_dtype dtype,
#                                   dsc_device_type device = DEFAULT,
#                                   dsc_data_buffer *buf = nullptr,
#                                   const void *DSC_RESTRICT data = nullptr,
#                                   dsc_device_type data_device = DEFAULT);
def _dsc_new_tensor(
    ctx: _DscCtx, shape: Tuple[int, ...], dtype: Dtype, device: Device 
) -> _DscTensor_p:
    shape_arr = (c_int * len(shape))(*shape)
    return _lib.dsc_new_tensor(ctx, len(shape), shape_arr, c_uint8(dtype.value), c_int8(device.value), None, None, c_int8(Device.DEFAULT.value))


_lib.dsc_new_tensor.argtypes = [_DscCtx, c_int, POINTER(c_int), c_uint8, c_int8, POINTER(_DscDataBuffer), c_void_p, c_int8]
_lib.dsc_new_tensor.restype = _DscTensor_p


# extern dsc_tensor *dsc_tensor_1d(dsc_ctx *ctx,
#                                  dsc_dtype dtype,
#                                  int dim1);
def _dsc_tensor_1d(
    ctx: _DscCtx, dtype: Dtype, dim1: int, device: Device, data: c_void_p, data_device: Device
) -> _DscTensor_p:
    return _lib.dsc_tensor_1d(ctx, c_uint8(dtype.value), c_int(dim1), c_int8(device.value), data, c_int8(data_device.value))


_lib.dsc_tensor_1d.argtypes = [_DscCtx, c_uint8, c_int, c_int8, c_void_p, c_int8]
_lib.dsc_tensor_1d.restype = _DscTensor_p


# extern dsc_tensor *dsc_tensor_2d(dsc_ctx *ctx,
#                                  dsc_dtype dtype,
#                                  int dim1, int dim2);
def _dsc_tensor_2d(
    ctx: _DscCtx,
    dtype: Dtype,
    dim1: int,
    dim2: int,
    device: Device,
    data: c_void_p,
    data_device: Device
) -> _DscTensor_p:
    return _lib.dsc_tensor_2d(
        ctx, c_uint8(dtype.value), c_int(dim1), c_int(dim2), c_int8(device.value), data, c_int8(data_device.value)
    )


_lib.dsc_tensor_2d.argtypes = [_DscCtx, c_uint8, c_int, c_int, c_int8, c_void_p, c_int8]
_lib.dsc_tensor_2d.restype = _DscTensor_p


# extern dsc_tensor *dsc_tensor_3d(dsc_ctx *ctx,
#                                  dsc_dtype dtype,
#                                  int dim1, int dim2,
#                                  int dim3);
def _dsc_tensor_3d(
    ctx: _DscCtx,
    dtype: Dtype,
    dim1: int,
    dim2: int,
    dim3: int,
    device: Device,
    data: c_void_p,
    data_device: Device
) -> _DscTensor_p:
    return _lib.dsc_tensor_3d(
        ctx, c_uint8(dtype.value), c_int(dim1), c_int(dim2), c_int(dim3), c_int8(device.value), data, c_int8(data_device.value)
    )


_lib.dsc_tensor_3d.argtypes = [_DscCtx, c_uint8, c_int, c_int, c_int, c_int8, c_void_p, c_int8]
_lib.dsc_tensor_3d.restype = _DscTensor_p


# extern dsc_tensor *dsc_tensor_4d(dsc_ctx *ctx,
#                                  dsc_dtype dtype,
#                                  int dim1, int dim2,
#                                  int dim3, int dim4);
def _dsc_tensor_4d(
    ctx: _DscCtx,
    dtype: Dtype,
    dim1: int,
    dim2: int,
    dim3: int,
    dim4: int,
    device: Device,
    data: c_void_p,
    data_device: Device
) -> _DscTensor_p:
    return _lib.dsc_tensor_4d(
        ctx,
        c_uint8(dtype.value),
        c_int(dim1),
        c_int(dim2),
        c_int(dim3),
        c_int(dim4),
        c_int8(device.value),
        data,
        c_int8(data_device.value)
    )


_lib.dsc_tensor_4d.argtypes = [_DscCtx, c_uint8, c_int, c_int, c_int, c_int, c_int8, c_void_p, c_int8]
_lib.dsc_tensor_4d.restype = _DscTensor_p


# extern dsc_tensor *dsc_wrap_bool(dsc_ctx *ctx,
#                                  bool val,
#                                  dsc_device_type device = DEFAULT);
def _dsc_wrap_bool(
    ctx: _DscCtx, val: bool, device: Device
) -> _DscTensor_p:
    return _lib.dsc_wrap_bool(ctx, c_bool(val), c_int8(device.value))


_lib.dsc_wrap_bool.argtypes = [_DscCtx, c_bool, c_int8]
_lib.dsc_wrap_bool.restype = _DscTensor_p


# extern dsc_tensor *dsc_wrap_i32(dsc_ctx *ctx,
#                                 i32 val,
#                                 dsc_device_type device = DEFAULT);
def _dsc_wrap_i32(
    ctx: _DscCtx, val: int, device: Device
) -> _DscTensor_p:
    return _lib.dsc_wrap_i32(ctx, c_int32(val), c_int8(device.value))


_lib.dsc_wrap_i32.argtypes = [_DscCtx, c_int32, c_int8]
_lib.dsc_wrap_i32.restype = _DscTensor_p


# extern dsc_tensor *dsc_wrap_f32(dsc_ctx *ctx,
#                                 f32 val,
#                                 dsc_device_type device = DEFAULT);
def _dsc_wrap_f32(
    ctx: _DscCtx, val: float, device: Device
) -> _DscTensor_p:
    return _lib.dsc_wrap_f32(ctx, c_float(val), c_int8(device.value))


_lib.dsc_wrap_f32.argtypes = [_DscCtx, c_float, c_int8]
_lib.dsc_wrap_f32.restype = _DscTensor_p


# extern dsc_tensor *dsc_wrap_f64(dsc_ctx *ctx,
#                                 f64 val,
#                                 dsc_device_type device = DEFAULT);
def _dsc_wrap_f64(
    ctx: _DscCtx, val: float, device: Device
) -> _DscTensor_p:
    return _lib.dsc_wrap_f64(ctx, c_double(val), c_int8(device.value))


_lib.dsc_wrap_f64.argtypes = [_DscCtx, c_double, c_int8]
_lib.dsc_wrap_f64.restype = _DscTensor_p


# extern dsc_tensor *dsc_arange(dsc_ctx *ctx,
#                               int n,
#                               dsc_dtype dtype = I32,
#                               dsc_device_type device = DEFAULT);
def _dsc_arange(
    ctx: _DscCtx, n: int, dtype: Dtype, device: Device
) -> _DscTensor_p:
    return _lib.dsc_arange(ctx, c_int(n), c_uint8(dtype.value), c_int8(device.value))


_lib.dsc_arange.argtypes = [_DscCtx, c_int, c_uint8, c_int8]
_lib.dsc_arange.restype = _DscTensor_p


# extern dsc_tensor *dsc_randn(dsc_ctx *ctx,
#                              int n_dim,
#                              const int *shape,
#                              dsc_dtype dtype = DSC_DEFAULT_TYPE);
def _dsc_randn(
    ctx: _DscCtx,
    shape: tuple[int, ...],
    dtype: Dtype,
    device: Device,
) -> _DscTensor_p:
    shape_type = c_int * len(shape)
    return _lib.dsc_randn(
        ctx, len(shape), shape_type(*shape), c_uint8(dtype.value), c_int8(device.value)
    )


_lib.dsc_randn.argtypes = [_DscCtx, c_int, POINTER(c_int), c_uint8, c_int8]
_lib.dsc_randn.restype = _DscTensor_p


# extern dsc_tensor *dsc_add(dsc_ctx *ctx,
#                            dsc_tensor *xa,
#                            dsc_tensor *xb,
#                            dsc_tensor *out = nullptr);
def _dsc_add(
    ctx: _DscCtx, xa: _DscTensor_p, xb: _DscTensor_p, out: _OptionalTensor
) -> _DscTensor_p:
    return _lib.dsc_add(ctx, xa, xb, out)


_lib.dsc_add.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, _DscTensor_p]
_lib.dsc_add.restype = _DscTensor_p


# extern dsc_tensor *dsc_sub(dsc_ctx *ctx,
#                            dsc_tensor *xa,
#                            dsc_tensor *xb,
#                            dsc_tensor *out = nullptr);
def _dsc_sub(
    ctx: _DscCtx, xa: _DscTensor_p, xb: _DscTensor_p, out: _OptionalTensor
) -> _DscTensor_p:
    return _lib.dsc_sub(ctx, xa, xb, out)


_lib.dsc_sub.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, _DscTensor_p]
_lib.dsc_sub.restype = _DscTensor_p


# extern dsc_tensor *dsc_mul(dsc_ctx *ctx,
#                            dsc_tensor *xa,
#                            dsc_tensor *xb,
#                            dsc_tensor *out = nullptr);
def _dsc_mul(
    ctx: _DscCtx, xa: _DscTensor_p, xb: _DscTensor_p, out: _OptionalTensor
) -> _DscTensor_p:
    return _lib.dsc_mul(ctx, xa, xb, out)


_lib.dsc_mul.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, _DscTensor_p]
_lib.dsc_mul.restype = _DscTensor_p


# extern dsc_tensor *dsc_div(dsc_ctx *ctx,
#                            dsc_tensor *xa,
#                            dsc_tensor *xb,
#                            dsc_tensor *out = nullptr);
def _dsc_div(
    ctx: _DscCtx, xa: _DscTensor_p, xb: _DscTensor_p, out: _OptionalTensor
) -> _DscTensor_p:
    return _lib.dsc_div(ctx, xa, xb, out)


_lib.dsc_div.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, _DscTensor_p]
_lib.dsc_div.restype = _DscTensor_p


# extern dsc_tensor *dsc_pow(dsc_ctx *ctx,
#                            dsc_tensor *xa,
#                            dsc_tensor *xb,
#                            dsc_tensor *out = nullptr);
def _dsc_pow(
    ctx: _DscCtx, xa: _DscTensor_p, xb: _DscTensor_p, out: _OptionalTensor
) -> _DscTensor_p:
    return _lib.dsc_pow(ctx, xa, xb, out)


_lib.dsc_pow.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, _DscTensor_p]
_lib.dsc_pow.restype = _DscTensor_p


# extern dsc_tensor *dsc_matmul(dsc_ctx *ctx,
#                               dsc_tensor *DSC_RESTRICT xa,
#                               dsc_tensor *DSC_RESTRICT xb,
#                               dsc_tensor *DSC_RESTRICT out = nullptr);
def _dsc_matmul(
    ctx: _DscCtx, xa: _DscTensor_p, xb: _DscTensor_p, out: _OptionalTensor
) -> _DscTensor_p:
    return _lib.dsc_matmul(ctx, xa, xb, out)


_lib.dsc_matmul.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, _DscTensor_p]
_lib.dsc_matmul.restype = _DscTensor_p


# extern dsc_tensor *dsc_compare(dsc_ctx *ctx,
#                                const dsc_tensor *xa,
#                                const dsc_tensor *xb,
#                                dsc_comparison_op comp,
#                                dsc_tensor *out = nullptr);
def _dsc_compare(
        ctx: _DscCtx, xa: _DscTensor_p, xb: _DscTensor_p, comp: _DscComparison, out: _OptionalTensor
) -> _DscTensor_p:
    return _lib.dsc_compare(ctx, xa, xb, c_uint8(comp.value), out)


_lib.dsc_compare.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, c_uint8, _DscTensor_p]
_lib.dsc_compare.restype = _DscTensor_p


# extern void dsc_masked_fill(dsc_ctx *ctx,
#                             dsc_tensor *x,
#                             const dsc_tensor *mask,
#                             f64 value);
def _dsc_masked_fill(
        ctx: _DscCtx, x: _DscTensor_p, mask: _DscTensor_p, value: float
):
    _lib.dsc_masked_fill(ctx, x, mask, c_double(value))


_lib.dsc_masked_fill.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, c_double]
_lib.dsc_masked_fill.restype = None


# extern dsc_tensor *dsc_cast(dsc_ctx *ctx,
#                             dsc_tensor *__restrict x,
#                             dsc_dtype new_dtype);
def _dsc_cast(ctx: _DscCtx, x: _DscTensor_p, dtype: Dtype) -> _DscTensor_p:
    return _lib.dsc_cast(ctx, x, c_uint8(dtype.value))


_lib.dsc_cast.argtypes = [_DscCtx, _DscTensor_p, c_uint8]
_lib.dsc_cast.restype = _DscTensor_p


# extern void dsc_copy(dsc_ctx *ctx,
#                      dsc_tensor *DSC_RESTRICT x,
#                      void *DSC_RESTRICT data,
#                      usize nb,
#                      dsc_device_type data_device = DEFAULT);
def _dsc_copy(ctx: _DscCtx, x: _DscTensor_p, data: c_void_p, nb: int, data_device: Device):
    _lib.dsc_copy(ctx, x, data, c_size_t(nb), c_int8(data_device.value))


_lib.dsc_copy.argtypes = [_DscCtx, _DscTensor_p, c_void_p, c_size_t, c_int8]
_lib.dsc_copy.restype = None


# extern dsc_tensor *dsc_reshape(dsc_ctx *ctx,
#                                const dsc_tensor *DSC_RESTRICT x,
#                                int dims...);
def _dsc_reshape(ctx: _DscCtx, x: _DscTensor_p, *dimensions: int) -> _DscTensor_p:
    return _lib.dsc_reshape(ctx, x, len(dimensions), *dimensions)


_lib.dsc_reshape.argtypes = [_DscCtx, _DscTensor_p, c_int]
_lib.dsc_reshape.restype = _DscTensor_p


# extern dsc_tensor *dsc_concat(dsc_ctx *ctx,
#                               int axis,
#                               int tensors...);
def _dsc_concat(ctx: _DscCtx, axis: int, *tensors: _DscTensor_p) -> _DscTensor_p:
    return _lib.dsc_concat(ctx, axis, len(tensors), *tensors)


_lib.dsc_concat.argtypes = [_DscCtx, c_int, c_int]
_lib.dsc_concat.restype = _DscTensor_p


# extern dsc_tensor *dsc_transpose(dsc_ctx *ctx,
#                                  const dsc_tensor *DSC_RESTRICT x,
#                                  int axes...) {
def _dsc_transpose(ctx: _DscCtx, x: _DscTensor_p, *axes: int) -> _DscTensor_p:
    return _lib.dsc_transpose(ctx, x, len(axes), *axes)


_lib.dsc_transpose.argtypes = [_DscCtx, _DscTensor_p, c_int]
_lib.dsc_transpose.restype = _DscTensor_p


# extern dsc_tensor *dsc_tril(dsc_ctx *ctx,
#                             const dsc_tensor *DSC_RESTRICT x,
#                             int diagonal = 0,
#                             dsc_tensor *DSC_RESTRICT out = nullptr);
def _dsc_tril(ctx: _DscCtx, x: _DscTensor_p, diagonal: int, out: _DscTensor_p) -> _DscTensor_p:
    return _lib.dsc_tril(ctx, x, c_int(diagonal), out)


_lib.dsc_tril.argtypes = [_DscCtx, _DscTensor_p, c_int, _DscTensor_p]
_lib.dsc_tril.restype = _DscTensor_p


# extern dsc_tensor *dsc_tensor_get_idx(dsc_ctx *ctx,
#                                       const dsc_tensor *DSC_RESTRICT x,
#                                       int indexes...);
def _dsc_tensor_get_idx(ctx: _DscCtx, x: _DscTensor_p, *indexes: int) -> _DscTensor_p:
    return _lib.dsc_tensor_get_idx(ctx, x, len(indexes), *indexes)


_lib.dsc_tensor_get_idx.argtypes = [_DscCtx, _DscTensor_p, c_int]
_lib.dsc_tensor_get_idx.restype = _DscTensor_p


# extern dsc_tensor *dsc_tensor_get_slice(dsc_ctx *ctx,
#                                         const dsc_tensor *DSC_RESTRICT x,
#                                         int slices...);
def _dsc_tensor_get_slice(
    ctx: _DscCtx, x: _DscTensor_p, *slices: _DscSlice
) -> _DscTensor_p:
    return _lib.dsc_tensor_get_slice(ctx, x, len(slices), *slices)


_lib.dsc_tensor_get_slice.argtypes = [_DscCtx, _DscTensor_p, c_int]
_lib.dsc_tensor_get_slice.restype = _DscTensor_p


# extern dsc_tensor *dsc_tensor_get_tensor(dsc_ctx *ctx,
#                                          const dsc_tensor *DSC_RESTRICT x,
#                                          const dsc_tensor *DSC_RESTRICT indexes);
def _dsc_tensor_get_tensor(
        ctx: _DscCtx, x: _DscTensor_p, indexes: _DscTensor_p
) -> _DscTensor_p:
    return _lib.dsc_tensor_get_tensor(ctx, x, indexes)


_lib.dsc_tensor_get_tensor.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p]
_lib.dsc_tensor_get_tensor.restype = _DscTensor_p


# extern void *dsc_tensor_set_idx(dsc_ctx *,
#                                 dsc_tensor *DSC_RESTRICT xa,
#                                 const dsc_tensor *DSC_RESTRICT xb,
#                                 int indexes...);
def _dsc_tensor_set_idx(
    ctx: _DscCtx, xa: _DscTensor_p, xb: _DscTensor_p, *indexes: int
):
    _lib.dsc_tensor_set_idx(ctx, xa, xb, len(indexes), *indexes)


_lib.dsc_tensor_set_idx.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, c_int]
_lib.dsc_tensor_set_idx.restype = None


# extern void *dsc_tensor_set_slice(dsc_ctx *,
#                                   dsc_tensor *DSC_RESTRICT xa,
#                                   const dsc_tensor *DSC_RESTRICT xb,
#                                   int slices...);
def _dsc_tensor_set_slice(
    ctx: _DscCtx, xa: _DscTensor_p, xb: _DscTensor_p, *slices: _DscSlice
):
    _lib.dsc_tensor_set_slice(ctx, xa, xb, len(slices), *slices)


_lib.dsc_tensor_set_slice.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, c_int]
_lib.dsc_tensor_set_slice.restype = None


# extern dsc_tensor *dsc_cos(dsc_ctx *,
#                            const dsc_tensor *__restrict x,
#                            dsc_tensor *__restrict out);
def _dsc_cos(ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor) -> _DscTensor_p:
    return _lib.dsc_cos(ctx, x, out)


_lib.dsc_cos.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p]
_lib.dsc_cos.restype = _DscTensor_p


# extern dsc_tensor *dsc_sin(dsc_ctx *,
#                            const dsc_tensor *__restrict x,
#                            dsc_tensor *__restrict out = nullptr);
def _dsc_sin(ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor) -> _DscTensor_p:
    return _lib.dsc_sin(ctx, x, out)


_lib.dsc_sin.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p]
_lib.dsc_sin.restype = _DscTensor_p


# extern dsc_tensor *dsc_tanh(dsc_ctx *,
#                             const dsc_tensor *__restrict x,
#                             dsc_tensor *__restrict out = nullptr);
def _dsc_tanh(ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor) -> _DscTensor_p:
    return _lib.dsc_tanh(ctx, x, out)


_lib.dsc_tanh.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p]
_lib.dsc_tanh.restype = _DscTensor_p


# extern dsc_tensor *dsc_exp(dsc_ctx *,
#                            const dsc_tensor *__restrict x,
#                            dsc_tensor *__restrict out = nullptr);
def _dsc_exp(ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor) -> _DscTensor_p:
    return _lib.dsc_exp(ctx, x, out)


_lib.dsc_exp.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p]
_lib.dsc_exp.restype = _DscTensor_p


# extern dsc_tensor *dsc_sqrt(dsc_ctx *,
#                             const dsc_tensor *__restrict x,
#                             dsc_tensor *__restrict out = nullptr);
def _dsc_sqrt(ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor) -> _DscTensor_p:
    return _lib.dsc_sqrt(ctx, x, out)


_lib.dsc_sqrt.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p]
_lib.dsc_sqrt.restype = _DscTensor_p


# extern dsc_tensor *dsc_sum(dsc_ctx *ctx,
#                            const dsc_tensor *DSC_RESTRICT x,
#                            dsc_tensor *DSC_RESTRICT out = nullptr,
#                            int axis = -1,
#                            bool keep_dims = true);
def _dsc_sum(
    ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor, axis: int, keepdims: bool
) -> _DscTensor_p:
    return _lib.dsc_sum(ctx, x, out, c_int(axis), c_bool(keepdims))


_lib.dsc_sum.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, c_int, c_bool]
_lib.dsc_sum.restype = _DscTensor_p


# extern dsc_tensor *dsc_max(dsc_ctx *ctx,
#                            const dsc_tensor *DSC_RESTRICT x,
#                            dsc_tensor *DSC_RESTRICT out = nullptr,
#                            int axis = -1,
#                            bool keep_dims = true);
def _dsc_max(
    ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor, axis: int, keepdims: bool
) -> _DscTensor_p:
    return _lib.dsc_max(ctx, x, out, c_int(axis), c_bool(keepdims))


_lib.dsc_max.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, c_int, c_bool]
_lib.dsc_max.restype = _DscTensor_p


# extern dsc_tensor *dsc_min(dsc_ctx *ctx,
#                            const dsc_tensor *DSC_RESTRICT x,
#                            dsc_tensor *DSC_RESTRICT out = nullptr,
#                            int axis = -1,
#                            bool keep_dims = true);
def _dsc_min(
    ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor, axis: int, keepdims: bool
) -> _DscTensor_p:
    return _lib.dsc_min(ctx, x, out, c_int(axis), c_bool(keepdims))


_lib.dsc_min.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, c_int, c_bool]
_lib.dsc_min.restype = _DscTensor_p
