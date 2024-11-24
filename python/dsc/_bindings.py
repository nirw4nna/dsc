# Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
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
    c_uint8,
    c_size_t,
    c_float,
    c_double,
    c_void_p,
    Structure,
    POINTER,
)
from typing import Union
from .dtype import Dtype


_DSC_MAX_DIMS = 4
_DSC_VALUE_NONE = 2**31 - 1

_DscCtx = c_void_p

# Todo: make this more flexible
_lib_file = f'{os.path.dirname(__file__)}/libdsc.so'
if not os.path.exists(_lib_file):
    raise RuntimeError(f'Error loading DSC shared object "{_lib_file}"')

_lib = ctypes.CDLL(_lib_file)


class _DscTensorBuffer(Structure):
    _fields_ = [
        ('refs', c_int),
    ]


class _DscTensor(Structure):
    _fields_ = [
        ('shape', c_int * _DSC_MAX_DIMS),
        ('stride', c_int * _DSC_MAX_DIMS),
        ('buffer', POINTER(_DscTensorBuffer)),
        ('data', c_void_p),
        ('ne', c_int),
        ('n_dim', c_int),
        ('dtype', c_uint8),
        ('backend', c_uint8),
    ]


_DscTensor_p_ = POINTER(_DscTensor)
# For some reason this format works fine with Pyright while just doing _DscTensor_p = POINTER(_DscTensor) doesn't
_DscTensor_p = _DscTensor_p_

_OptionalTensor = Union[_DscTensor_p, None]


class _C32(Structure):
    _fields_ = [('real', c_float), ('imag', c_float)]


class _C64(Structure):
    _fields_ = [('real', c_double), ('imag', c_double)]


class _DscSlice(Structure):
    _fields_ = [('start', c_int), ('stop', c_int), ('step', c_int)]


# extern dsc_ctx *dsc_ctx_init(usize main_mem, usize scratch_mem) noexcept;
def _dsc_ctx_init(main_mem: int, scratch_mem: int) -> _DscCtx:
    return _lib.dsc_ctx_init(c_size_t(main_mem), c_size_t(scratch_mem))


_lib.dsc_ctx_init.argtypes = [c_size_t, c_size_t]
_lib.dsc_ctx_init.restype = _DscCtx


# extern dsc_fft_plan *dsc_plan_fft(dsc_ctx *ctx,
#                                   const int n,
#                                   const dsc_dtype dtype) noexcept;
def _dsc_plan_fft(ctx: _DscCtx, n: int, dtype: Dtype):
    return _lib.dsc_plan_fft(ctx, c_int(n), c_uint8(dtype.value))


_lib.dsc_plan_fft.argtypes = [_DscCtx, c_int, c_uint8]
_lib.dsc_plan_fft.restype = c_void_p


# extern void dsc_ctx_free(dsc_ctx *ctx) noexcept;
def _dsc_ctx_free(ctx: _DscCtx):
    _lib.dsc_ctx_free(ctx)


_lib.dsc_ctx_free.argtypes = [_DscCtx]
_lib.dsc_ctx_free.restype = None


# extern void dsc_ctx_clear(dsc_ctx *ctx) noexcept;
def _dsc_ctx_clear(ctx: _DscCtx):
    _lib.dsc_ctx_clear(ctx)


_lib.dsc_ctx_clear.argtypes = [_DscCtx]
_lib.dsc_ctx_clear.restype = None


# extern void dsc_tensor_free(dsc_ctx *ctx, dsc_tensor *x) noexcept;
def _dsc_tensor_free(ctx: _DscCtx, x: _DscTensor_p):
    _lib.dsc_tensor_free(ctx, x)


_lib.dsc_tensor_free.argtypes = [_DscCtx, _DscTensor_p]
_lib.dsc_tensor_free.restype = None


# extern usize dsc_used_mem(dsc_ctx *ctx) noexcept;
def _dsc_used_mem(ctx: _DscCtx) -> int:
    return _lib.dsc_used_mem(ctx)


_lib.dsc_used_mem.argtypes = [_DscCtx]
_lib.dsc_used_mem.restype = c_size_t


# extern void dsc_print_mem_usage(dsc_ctx *ctx) noexcept;
def _dsc_print_mem_usage(ctx: _DscCtx):
    _lib.dsc_print_mem_usage(ctx)


_lib.dsc_print_mem_usage.argtypes = [_DscCtx]
_lib.dsc_print_mem_usage.restype = None


# extern void dsc_traces_record(dsc_ctx *ctx, bool record) noexcept;
def _dsc_traces_record(ctx: _DscCtx, record: bool):
    _lib.dsc_traces_record(ctx, c_bool(record))


_lib.dsc_traces_record.argtypes = [_DscCtx, c_bool]
_lib.dsc_traces_record.restype = None


# extern void dsc_dump_traces(dsc_ctx *ctx, const char *filename) noexcept;
def _dsc_dump_traces(ctx: _DscCtx, filename: str):
    _lib.dsc_dump_traces(ctx, c_char_p(filename.encode('utf-8')))


_lib.dsc_dump_traces.argtypes = [_DscCtx, c_char_p]
_lib.dsc_dump_traces.restype = None


# extern void dsc_clear_traces(dsc_ctx *) noexcept;
def _dsc_clear_traces(ctx: _DscCtx):
    _lib.dsc_clear_traces(ctx)


_lib.dsc_clear_traces.argtypes = [_DscCtx]
_lib.dsc_clear_traces.restype = None


# extern dsc_tensor *dsc_view(dsc_ctx *ctx,
#                             const dsc_tensor *x) noexcept;
def _dsc_view(ctx: _DscCtx, x: _DscTensor_p) -> _DscTensor_p:
    return _lib.dsc_view(ctx, x)


_lib.dsc_view.argtypes = [_DscCtx, _DscTensor_p]
_lib.dsc_view.restype = _DscTensor_p


# extern dsc_tensor *dsc_tensor_1d(dsc_ctx *ctx,
#                                  dsc_dtype dtype,
#                                  int dim1) noexcept;
def _dsc_tensor_1d(ctx: _DscCtx, dtype: Dtype, dim1: int) -> _DscTensor_p:
    return _lib.dsc_tensor_1d(ctx, c_uint8(dtype.value), c_int(dim1))


_lib.dsc_tensor_1d.argtypes = [_DscCtx, c_uint8, c_int]
_lib.dsc_tensor_1d.restype = _DscTensor_p


# extern dsc_tensor *dsc_tensor_2d(dsc_ctx *ctx,
#                                  dsc_dtype dtype,
#                                  int dim1, int dim2) noexcept;
def _dsc_tensor_2d(ctx: _DscCtx, dtype: Dtype, dim1: int, dim2: int) -> _DscTensor_p:
    return _lib.dsc_tensor_2d(ctx, c_uint8(dtype.value), c_int(dim1), c_int(dim2))


_lib.dsc_tensor_2d.argtypes = [_DscCtx, c_uint8, c_int, c_int]
_lib.dsc_tensor_2d.restype = _DscTensor_p


# extern dsc_tensor *dsc_tensor_3d(dsc_ctx *ctx,
#                                  dsc_dtype dtype,
#                                  int dim1, int dim2,
#                                  int dim3) noexcept;
def _dsc_tensor_3d(
    ctx: _DscCtx, dtype: Dtype, dim1: int, dim2: int, dim3: int
) -> _DscTensor_p:
    return _lib.dsc_tensor_3d(
        ctx, c_uint8(dtype.value), c_int(dim1), c_int(dim2), c_int(dim3)
    )


_lib.dsc_tensor_3d.argtypes = [_DscCtx, c_uint8, c_int, c_int, c_int]
_lib.dsc_tensor_3d.restype = _DscTensor_p


# extern dsc_tensor *dsc_tensor_4d(dsc_ctx *ctx,
#                                  dsc_dtype dtype,
#                                  int dim1, int dim2,
#                                  int dim3, int dim4) noexcept;
def _dsc_tensor_4d(
    ctx: _DscCtx, dtype: Dtype, dim1: int, dim2: int, dim3: int, dim4: int
) -> _DscTensor_p:
    return _lib.dsc_tensor_4d(
        ctx, c_uint8(dtype.value), c_int(dim1), c_int(dim2), c_int(dim3), c_int(dim4)
    )


_lib.dsc_tensor_4d.argtypes = [_DscCtx, c_uint8, c_int, c_int, c_int, c_int]
_lib.dsc_tensor_4d.restype = _DscTensor_p


# extern dsc_tensor *dsc_wrap_f32(dsc_ctx *ctx,
#                                 f32 val) noexcept;
def _dsc_wrap_f32(ctx: _DscCtx, val: float) -> _DscTensor_p:
    return _lib.dsc_wrap_f32(ctx, c_float(val))


_lib.dsc_wrap_f32.argtypes = [_DscCtx, c_float]
_lib.dsc_wrap_f32.restype = _DscTensor_p


# extern dsc_tensor *dsc_wrap_f64(dsc_ctx *ctx,
#                                 f64 val) noexcept;
def _dsc_wrap_f64(ctx: _DscCtx, val: float) -> _DscTensor_p:
    return _lib.dsc_wrap_f64(ctx, c_double(val))


_lib.dsc_wrap_f64.argtypes = [_DscCtx, c_double]
_lib.dsc_wrap_f64.restype = _DscTensor_p


# extern dsc_tensor *dsc_wrap_c32(dsc_ctx *ctx,
#                                 c32 val) noexcept;
def _dsc_wrap_c32(ctx: _DscCtx, val: complex) -> _DscTensor_p:
    return _lib.dsc_wrap_c32(ctx, _C32(c_float(val.real), c_float(val.imag)))


_lib.dsc_wrap_c32.argtypes = [_DscCtx, _C32]
_lib.dsc_wrap_c32.restype = _DscTensor_p


# extern dsc_tensor *dsc_wrap_c64(dsc_ctx *ctx,
#                                 c64 val) noexcept;
def _dsc_wrap_c64(ctx: _DscCtx, val: complex) -> _DscTensor_p:
    return _lib.dsc_wrap_c64(ctx, _C64(c_double(val.real), c_double(val.imag)))


_lib.dsc_wrap_c64.argtypes = [_DscCtx, _C64]
_lib.dsc_wrap_c64.restype = _DscTensor_p


# extern dsc_tensor *dsc_arange(dsc_ctx *ctx,
#                               int n,
#                               dsc_dtype dtype = DSC_DEFAULT_TYPE) noexcept;
def _dsc_arange(ctx: _DscCtx, n: int, dtype: Dtype) -> _DscTensor_p:
    return _lib.dsc_arange(ctx, c_int(n), c_uint8(dtype.value))


_lib.dsc_arange.argtypes = [_DscCtx, c_int, c_uint8]
_lib.dsc_arange.restype = _DscTensor_p


# extern dsc_tensor *dsc_randn(dsc_ctx *ctx,
#                              int n_dim,
#                              const int *shape,
#                              dsc_dtype dtype = DSC_DEFAULT_TYPE) noexcept;
def _dsc_randn(ctx: _DscCtx, shape: tuple[int, ...], dtype: Dtype) -> _DscTensor_p:
    shape_type = c_int * len(shape)
    return _lib.dsc_randn(ctx, len(shape), shape_type(*shape), c_uint8(dtype.value))


_lib.dsc_randn.argtypes = [_DscCtx, c_int, POINTER(c_int), c_uint8]
_lib.dsc_randn.restype = _DscTensor_p


# extern dsc_tensor *dsc_add(dsc_ctx *ctx,
#                            dsc_tensor *xa,
#                            dsc_tensor *xb,
#                            dsc_tensor *out = nullptr) noexcept;
def _dsc_add(
    ctx: _DscCtx, xa: _DscTensor_p, xb: _DscTensor_p, out: _OptionalTensor
) -> _DscTensor_p:
    return _lib.dsc_add(ctx, xa, xb, out)


_lib.dsc_add.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, _DscTensor_p]
_lib.dsc_add.restype = _DscTensor_p


# extern dsc_tensor *dsc_sub(dsc_ctx *ctx,
#                            dsc_tensor *xa,
#                            dsc_tensor *xb,
#                            dsc_tensor *out = nullptr) noexcept;
def _dsc_sub(
    ctx: _DscCtx, xa: _DscTensor_p, xb: _DscTensor_p, out: _OptionalTensor
) -> _DscTensor_p:
    return _lib.dsc_sub(ctx, xa, xb, out)


_lib.dsc_sub.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, _DscTensor_p]
_lib.dsc_sub.restype = _DscTensor_p


# extern dsc_tensor *dsc_mul(dsc_ctx *ctx,
#                            dsc_tensor *xa,
#                            dsc_tensor *xb,
#                            dsc_tensor *out = nullptr) noexcept;
def _dsc_mul(
    ctx: _DscCtx, xa: _DscTensor_p, xb: _DscTensor_p, out: _OptionalTensor
) -> _DscTensor_p:
    return _lib.dsc_mul(ctx, xa, xb, out)


_lib.dsc_mul.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, _DscTensor_p]
_lib.dsc_mul.restype = _DscTensor_p


# extern dsc_tensor *dsc_div(dsc_ctx *ctx,
#                            dsc_tensor *xa,
#                            dsc_tensor *xb,
#                            dsc_tensor *out = nullptr) noexcept;
def _dsc_div(
    ctx: _DscCtx, xa: _DscTensor_p, xb: _DscTensor_p, out: _OptionalTensor
) -> _DscTensor_p:
    return _lib.dsc_div(ctx, xa, xb, out)


_lib.dsc_div.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, _DscTensor_p]
_lib.dsc_div.restype = _DscTensor_p


# extern dsc_tensor *dsc_pow(dsc_ctx *ctx,
#                            dsc_tensor *xa,
#                            dsc_tensor *xb,
#                            dsc_tensor *out = nullptr) noexcept;
def _dsc_pow(
    ctx: _DscCtx, xa: _DscTensor_p, xb: _DscTensor_p, out: _OptionalTensor
) -> _DscTensor_p:
    return _lib.dsc_pow(ctx, xa, xb, out)


_lib.dsc_pow.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, _DscTensor_p]
_lib.dsc_pow.restype = _DscTensor_p


# extern dsc_tensor *dsc_cast(dsc_ctx *ctx,
#                             dsc_tensor *__restrict x,
#                             dsc_dtype new_dtype) noexcept;
def _dsc_cast(ctx: _DscCtx, x: _DscTensor_p, dtype: Dtype) -> _DscTensor_p:
    return _lib.dsc_cast(ctx, x, c_uint8(dtype.value))


_lib.dsc_cast.argtypes = [_DscCtx, _DscTensor_p, c_uint8]
_lib.dsc_cast.restype = _DscTensor_p


# extern dsc_tensor *dsc_reshape(dsc_ctx *ctx,
#                                const dsc_tensor *DSC_RESTRICT x,
#                                int dims...) noexcept;
def _dsc_reshape(ctx: _DscCtx, x: _DscTensor_p, *dimensions: int) -> _DscTensor_p:
    return _lib.dsc_reshape(ctx, x, len(dimensions), *dimensions)


_lib.dsc_reshape.argtypes = [_DscCtx, _DscTensor_p, c_int]
_lib.dsc_reshape.restype = _DscTensor_p


# extern dsc_tensor *dsc_concat(dsc_ctx *ctx,
#                               int axis,
#                               int tensors...) noexcept;
def _dsc_concat(ctx: _DscCtx, axis: int, *tensors: _DscTensor_p) -> _DscTensor_p:
    return _lib.dsc_concat(ctx, axis, len(tensors), *tensors)


_lib.dsc_concat.argtypes = [_DscCtx, c_int, c_int]
_lib.dsc_concat.restype = _DscTensor_p


# extern dsc_tensor *dsc_transpose(dsc_ctx *ctx,
#                                  const dsc_tensor *DSC_RESTRICT x,
#                                  int axes...) noexcept {
def _dsc_transpose(ctx: _DscCtx, x: _DscTensor_p, *axes: int) -> _DscTensor_p:
    return _lib.dsc_transpose(ctx, x, len(axes), *axes)


_lib.dsc_transpose.argtypes = [_DscCtx, _DscTensor_p, c_int]
_lib.dsc_transpose.restype = _DscTensor_p


# extern dsc_tensor *dsc_tensor_get_idx(dsc_ctx *ctx,
#                                       const dsc_tensor *DSC_RESTRICT x,
#                                       int indexes...) noexcept;
def _dsc_tensor_get_idx(ctx: _DscCtx, x: _DscTensor_p, *indexes: int) -> _DscTensor_p:
    return _lib.dsc_tensor_get_idx(ctx, x, len(indexes), *indexes)


_lib.dsc_tensor_get_idx.argtypes = [_DscCtx, _DscTensor_p, c_int]
_lib.dsc_tensor_get_idx.restype = _DscTensor_p


# extern dsc_tensor *dsc_tensor_get_slice(dsc_ctx *ctx,
#                                         const dsc_tensor *DSC_RESTRICT x,
#                                         int slices...) noexcept;
def _dsc_tensor_get_slice(
    ctx: _DscCtx, x: _DscTensor_p, *slices: _DscSlice
) -> _DscTensor_p:
    return _lib.dsc_tensor_get_slice(ctx, x, len(slices), *slices)


_lib.dsc_tensor_get_slice.argtypes = [_DscCtx, _DscTensor_p, c_int]
_lib.dsc_tensor_get_slice.restype = _DscTensor_p


# extern void *dsc_tensor_set_idx(dsc_ctx *,
#                                 dsc_tensor *DSC_RESTRICT xa,
#                                 const dsc_tensor *DSC_RESTRICT xb,
#                                 int indexes...) noexcept;
def _dsc_tensor_set_idx(
    ctx: _DscCtx, xa: _DscTensor_p, xb: _DscTensor_p, *indexes: int
):
    _lib.dsc_tensor_set_idx(ctx, xa, xb, len(indexes), *indexes)


_lib.dsc_tensor_set_idx.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, c_int]
_lib.dsc_tensor_set_idx.restype = None


# extern void *dsc_tensor_set_slice(dsc_ctx *,
#                                   dsc_tensor *DSC_RESTRICT xa,
#                                   const dsc_tensor *DSC_RESTRICT xb,
#                                   int slices...) noexcept;
def _dsc_tensor_set_slice(
    ctx: _DscCtx, xa: _DscTensor_p, xb: _DscTensor_p, *slices: _DscSlice
):
    _lib.dsc_tensor_set_slice(ctx, xa, xb, len(slices), *slices)


_lib.dsc_tensor_set_slice.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, c_int]
_lib.dsc_tensor_set_slice.restype = None


# extern dsc_tensor *dsc_cos(dsc_ctx *,
#                            const dsc_tensor *__restrict x,
#                            dsc_tensor *__restrict out) noexcept;
def _dsc_cos(ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor) -> _DscTensor_p:
    return _lib.dsc_cos(ctx, x, out)


_lib.dsc_cos.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p]
_lib.dsc_cos.restype = _DscTensor_p


# extern dsc_tensor *dsc_sin(dsc_ctx *,
#                            const dsc_tensor *__restrict x,
#                            dsc_tensor *__restrict out = nullptr) noexcept;
def _dsc_sin(ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor) -> _DscTensor_p:
    return _lib.dsc_sin(ctx, x, out)


_lib.dsc_sin.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p]
_lib.dsc_sin.restype = _DscTensor_p


# extern dsc_tensor *dsc_sinc(dsc_ctx *,
#                             const dsc_tensor *__restrict x,
#                             dsc_tensor *__restrict out = nullptr) noexcept;
def _dsc_sinc(ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor) -> _DscTensor_p:
    return _lib.dsc_sinc(ctx, x, out)


_lib.dsc_sinc.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p]
_lib.dsc_sinc.restype = _DscTensor_p


# extern dsc_tensor *dsc_logn(dsc_ctx *,
#                             const dsc_tensor *__restrict x,
#                             dsc_tensor *__restrict out = nullptr) noexcept;
def _dsc_logn(ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor) -> _DscTensor_p:
    return _lib.dsc_logn(ctx, x, out)


_lib.dsc_logn.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p]
_lib.dsc_logn.restype = _DscTensor_p


# extern dsc_tensor *dsc_log2(dsc_ctx *,
#                             const dsc_tensor *__restrict x,
#                             dsc_tensor *__restrict out = nullptr) noexcept;
def _dsc_log2(ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor) -> _DscTensor_p:
    return _lib.dsc_log2(ctx, x, out)


_lib.dsc_log2.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p]
_lib.dsc_log2.restype = _DscTensor_p


# extern dsc_tensor *dsc_log10(dsc_ctx *,
#                              const dsc_tensor *__restrict x,
#                              dsc_tensor *__restrict out = nullptr) noexcept;
def _dsc_log10(ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor) -> _DscTensor_p:
    return _lib.dsc_log10(ctx, x, out)


_lib.dsc_log10.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p]
_lib.dsc_log10.restype = _DscTensor_p


# extern dsc_tensor *dsc_exp(dsc_ctx *,
#                            const dsc_tensor *__restrict x,
#                            dsc_tensor *__restrict out = nullptr) noexcept;
def _dsc_exp(ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor) -> _DscTensor_p:
    return _lib.dsc_exp(ctx, x, out)


_lib.dsc_exp.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p]
_lib.dsc_exp.restype = _DscTensor_p


# extern dsc_tensor *dsc_sqrt(dsc_ctx *,
#                             const dsc_tensor *__restrict x,
#                             dsc_tensor *__restrict out = nullptr) noexcept;
def _dsc_sqrt(ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor) -> _DscTensor_p:
    return _lib.dsc_sqrt(ctx, x, out)


_lib.dsc_sqrt.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p]
_lib.dsc_sqrt.restype = _DscTensor_p


# extern dsc_tensor *dsc_abs(dsc_ctx *,
#                             const dsc_tensor *__restrict x,
#                             dsc_tensor *__restrict out = nullptr) noexcept;
def _dsc_abs(ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor) -> _DscTensor_p:
    return _lib.dsc_abs(ctx, x, out)


_lib.dsc_abs.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p]
_lib.dsc_abs.restype = _DscTensor_p


# extern dsc_tensor *dsc_angle(dsc_ctx *,
#                              const dsc_tensor *__restrict x) noexcept;
def _dsc_angle(ctx: _DscCtx, x: _DscTensor_p) -> _DscTensor_p:
    return _lib.dsc_angle(ctx, x)


_lib.dsc_angle.argtypes = [_DscCtx, _DscTensor_p]
_lib.dsc_angle.restype = _DscTensor_p


# extern dsc_tensor *dsc_conj(dsc_ctx *,
#                             dsc_tensor *__restrict x) noexcept;
def _dsc_conj(ctx: _DscCtx, x: _DscTensor_p) -> _DscTensor_p:
    return _lib.dsc_conj(ctx, x)


_lib.dsc_conj.argtypes = [_DscCtx, _DscTensor_p]
_lib.dsc_conj.restype = _DscTensor_p


# extern dsc_tensor *dsc_real(dsc_ctx *,
#                             dsc_tensor *__restrict x) noexcept;
def _dsc_real(ctx: _DscCtx, x: _DscTensor_p) -> _DscTensor_p:
    return _lib.dsc_real(ctx, x)


_lib.dsc_real.argtypes = [_DscCtx, _DscTensor_p]
_lib.dsc_real.restype = _DscTensor_p


# extern dsc_tensor *dsc_imag(dsc_ctx *,
#                             const dsc_tensor *__restrict x) noexcept;
def _dsc_imag(ctx: _DscCtx, x: _DscTensor_p) -> _DscTensor_p:
    return _lib.dsc_imag(ctx, x)


_lib.dsc_imag.argtypes = [_DscCtx, _DscTensor_p]
_lib.dsc_imag.restype = _DscTensor_p


# extern dsc_tensor *dsc_i0(dsc_ctx *,
#                           const dsc_tensor *__restrict x) noexcept;
def _dsc_i0(ctx: _DscCtx, x: _DscTensor_p) -> _DscTensor_p:
    return _lib.dsc_i0(ctx, x)


_lib.dsc_i0.argtypes = [_DscCtx, _DscTensor_p]
_lib.dsc_i0.restype = _DscTensor_p


# extern dsc_tensor *dsc_clip(dsc_ctx *ctx,
#                             const dsc_tensor *DSC_RESTRICT x,
#                             dsc_tensor *DSC_RESTRICT out = nullptr,
#                             f64 x_min = dsc_inf<f64, false>(),
#                             f64 x_max = dsc_inf<f64, true>()) noexcept;
def _dsc_clip(
    ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor, x_min: float, x_max: float
) -> _DscTensor_p:
    return _lib.dsc_clip(ctx, x, out, c_double(x_min), c_double(x_max))


_lib.dsc_clip.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, c_double, c_double]
_lib.dsc_clip.restype = _DscTensor_p


# extern dsc_tensor *dsc_sum(dsc_ctx *ctx,
#                            const dsc_tensor *DSC_RESTRICT x,
#                            dsc_tensor *DSC_RESTRICT out = nullptr,
#                            int axis = -1,
#                            bool keep_dims = true) noexcept;
def _dsc_sum(
    ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor, axis: int, keepdims: bool
) -> _DscTensor_p:
    return _lib.dsc_sum(ctx, x, out, c_int(axis), c_bool(keepdims))


_lib.dsc_sum.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, c_int, c_bool]
_lib.dsc_sum.restype = _DscTensor_p


# extern dsc_tensor *dsc_mean(dsc_ctx *ctx,
#                            const dsc_tensor *DSC_RESTRICT x,
#                            dsc_tensor *DSC_RESTRICT out = nullptr,
#                            int axis = -1,
#                            bool keep_dims = true) noexcept;
def _dsc_mean(
    ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor, axis: int, keepdims: bool
) -> _DscTensor_p:
    return _lib.dsc_mean(ctx, x, out, c_int(axis), c_bool(keepdims))


_lib.dsc_mean.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, c_int, c_bool]
_lib.dsc_mean.restype = _DscTensor_p


# extern dsc_tensor *dsc_max(dsc_ctx *ctx,
#                            const dsc_tensor *DSC_RESTRICT x,
#                            dsc_tensor *DSC_RESTRICT out = nullptr,
#                            int axis = -1,
#                            bool keep_dims = true) noexcept;
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
#                            bool keep_dims = true) noexcept;
def _dsc_min(
    ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor, axis: int, keepdims: bool
) -> _DscTensor_p:
    return _lib.dsc_min(ctx, x, out, c_int(axis), c_bool(keepdims))


_lib.dsc_min.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, c_int, c_bool]
_lib.dsc_min.restype = _DscTensor_p


# extern dsc_tensor *dsc_fft(dsc_ctx *ctx,
#                            const dsc_tensor *DSC_RESTRICT x,
#                            dsc_tensor *DSC_RESTRICT out,
#                            int n = -1,
#                            int axis = -1) noexcept;
def _dsc_fft(
    ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor, n: int, axis: int
) -> _DscTensor_p:
    return _lib.dsc_fft(ctx, x, out, c_int(n), c_int(axis))


_lib.dsc_fft.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, c_int, c_int]
_lib.dsc_fft.restype = _DscTensor_p


# extern dsc_tensor *dsc_ifft(dsc_ctx *ctx,
#                             const dsc_tensor *DSC_RESTRICT x,
#                             dsc_tensor *DSC_RESTRICT out,
#                             int n = -1,
#                             int axis = -1) noexcept;
def _dsc_ifft(
    ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor, n: int, axis: int
) -> _DscTensor_p:
    return _lib.dsc_ifft(ctx, x, out, c_int(n), c_int(axis))


_lib.dsc_ifft.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, c_int, c_int]
_lib.dsc_ifft.restype = _DscTensor_p


# extern dsc_tensor *dsc_rfft(dsc_ctx *ctx,
#                             const dsc_tensor *DSC_RESTRICT x,
#                             dsc_tensor *DSC_RESTRICT out,
#                             int n = -1,
#                             int axis = -1) noexcept;
def _dsc_rfft(
    ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor, n: int, axis: int
) -> _DscTensor_p:
    return _lib.dsc_rfft(ctx, x, out, c_int(n), c_int(axis))


_lib.dsc_rfft.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, c_int, c_int]
_lib.dsc_rfft.restype = _DscTensor_p


# extern dsc_tensor *dsc_irfft(dsc_ctx *ctx,
#                              const dsc_tensor *DSC_RESTRICT x,
#                              dsc_tensor *DSC_RESTRICT out,
#                              int n = -1,
#                              int axis = -1) noexcept;
def _dsc_irfft(
    ctx: _DscCtx, x: _DscTensor_p, out: _OptionalTensor, n: int, axis: int
) -> _DscTensor_p:
    return _lib.dsc_irfft(ctx, x, out, c_int(n), c_int(axis))


_lib.dsc_irfft.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, c_int, c_int]
_lib.dsc_irfft.restype = _DscTensor_p


# extern dsc_tensor *dsc_fftfreq(dsc_ctx *ctx,
#                                int n,
#                                f64 d = 1.,
#                                dsc_dtype dtype = DSC_DEFAULT_TYPE) noexcept;
def _dsc_fftfreq(ctx: _DscCtx, n: int, d: float, dtype: Dtype) -> _DscTensor_p:
    return _lib.dsc_fftfreq(ctx, c_int(n), c_double(d), c_uint8(dtype.value))


_lib.dsc_fftfreq.argtypes = [_DscCtx, c_int, c_double, c_uint8]
_lib.dsc_fftfreq.restype = _DscTensor_p


# extern dsc_tensor *dsc_rfftfreq(dsc_ctx *ctx,
#                                 int n,
#                                 f64 d = 1.,
#                                 dsc_dtype dtype = DSC_DEFAULT_TYPE) noexcept;
def _dsc_rfftfreq(ctx: _DscCtx, n: int, d: float, dtype: Dtype) -> _DscTensor_p:
    return _lib.dsc_rfftfreq(ctx, c_int(n), c_double(d), c_uint8(dtype.value))


_lib.dsc_rfftfreq.argtypes = [_DscCtx, c_int, c_double, c_uint8]
_lib.dsc_rfftfreq.restype = _DscTensor_p
