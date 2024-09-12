# Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
# All rights reserved.
#
# This code is licensed under the terms of the 3-clause BSD license
# (https://opensource.org/license/bsd-3-clause).

import os
import ctypes
from ctypes import (
    c_bool,
    c_char,
    c_char_p,
    c_int,
    c_int8,
    c_int32,
    c_int64,
    c_uint8,
    c_uint32,
    c_size_t,
    c_float,
    c_double,
    c_void_p,
    Structure,
    Array,
    POINTER
)

_DSC_MAX_DIMS = 4

_DscCtx = c_void_p

# Todo: make this more flexible
_lib_file = f'{os.path.dirname(__file__)}/libdsc.so'
if not os.path.exists(_lib_file):
    raise RuntimeError(f'Error loading DSC shared object "{_lib_file}"')

_lib = ctypes.CDLL(_lib_file)


class _DscTensor(Structure):
    _fields_ = [
        ('shape', c_int * _DSC_MAX_DIMS),
        ('stride', c_int * _DSC_MAX_DIMS),
        ('data', c_void_p),
        ('ne', c_int),
        ('n_dim', c_int),
        ('dtype', c_uint8),
    ]


_DscTensor_p = POINTER(_DscTensor)


class _C32(Structure):
    _fields_ = [('real', c_float), ('imag', c_float)]


class _C64(Structure):
    _fields_ = [('real', c_double), ('imag', c_double)]


# extern dsc_ctx *dsc_ctx_init(usize nb) noexcept;
def _dsc_ctx_init(nb: int) -> _DscCtx:
    return _lib.dsc_ctx_init(c_size_t(nb))


_lib.dsc_ctx_init.argtypes = [c_size_t]
_lib.dsc_ctx_init.restype = _DscCtx


# extern dsc_fft_plan dsc_plan_fft(dsc_ctx *ctx,
#                                  const int n,
#                                  const dsc_dtype dtype) noexcept;
def _dsc_plan_fft(ctx: _DscCtx, n: int, dtype: c_uint8):
    return _lib.dsc_plan_fft(ctx, n, dtype)


_lib.dsc_plan_fft.argtypes = [_DscCtx, c_int, c_uint8]
_lib.dsc_plan_fft.restype = c_void_p


# extern void dsc_ctx_free(dsc_ctx *ctx) noexcept;
def _dsc_ctx_free(ctx: _DscCtx):
    return _lib.dsc_ctx_free(ctx)


_lib.dsc_ctx_free.argtypes = [_DscCtx]
_lib.dsc_ctx_free.restype = None


# extern void dsc_ctx_clear(dsc_ctx *ctx) noexcept;
def _dsc_ctx_clear(ctx: _DscCtx):
    return _lib.dsc_ctx_clear(ctx)


_lib.dsc_ctx_clear.argtypes = [_DscCtx]
_lib.dsc_ctx_clear.restype = None


# extern dsc_tensor *dsc_tensor_1d(dsc_ctx *ctx,
#                                  dsc_dtype dtype,
#                                  int dim1) noexcept;
def _dsc_tensor_1d(ctx: _DscCtx, dtype: c_uint8, dim1: c_int):
    return _lib.dsc_tensor_1d(ctx, dtype, dim1)


_lib.dsc_tensor_1d.argtypes = [_DscCtx, c_uint8, c_int]
_lib.dsc_tensor_1d.restype = _DscTensor_p


# extern dsc_tensor *dsc_tensor_2d(dsc_ctx *ctx,
#                                  dsc_dtype dtype,
#                                  int dim1, int dim2) noexcept;
def _dsc_tensor_2d(ctx: _DscCtx, dtype: c_uint8, dim1: c_int, dim2: c_int):
    return _lib.dsc_tensor_2d(ctx, dtype, dim1, dim2)


_lib.dsc_tensor_2d.argtypes = [_DscCtx, c_uint8, c_int, c_int]
_lib.dsc_tensor_2d.restype = _DscTensor_p


# extern dsc_tensor *dsc_tensor_3d(dsc_ctx *ctx,
#                                  dsc_dtype dtype,
#                                  int dim1, int dim2,
#                                  int dim3) noexcept;
def _dsc_tensor_3d(ctx: _DscCtx, dtype: c_uint8, dim1: c_int, dim2: c_int, dim3: c_int):
    return _lib.dsc_tensor_3d(ctx, dtype, dim1, dim2, dim3)


_lib.dsc_tensor_3d.argtypes = [_DscCtx, c_uint8, c_int, c_int, c_int]
_lib.dsc_tensor_3d.restype = _DscTensor_p


# extern dsc_tensor *dsc_tensor_4d(dsc_ctx *ctx,
#                                  dsc_dtype dtype,
#                                  int dim1, int dim2,
#                                  int dim3, int dim4) noexcept;
def _dsc_tensor_4d(ctx: _DscCtx, dtype: c_uint8,
                   dim1: c_int, dim2: c_int,
                   dim3: c_int, dim4: c_int):
    return _lib.dsc_tensor_4d(ctx, dtype, dim1, dim2, dim3, dim4)


_lib.dsc_tensor_4d.argtypes = [_DscCtx, c_uint8, c_int, c_int, c_int, c_int]
_lib.dsc_tensor_4d.restype = _DscTensor_p


# extern dsc_tensor *dsc_arange(dsc_ctx *ctx,
#                               int n,
#                               dsc_dtype dtype = DSC_DEFAULT_TYPE) noexcept;
def _dsc_arange(ctx: _DscCtx, n: int, dtype: c_uint8) -> _DscTensor_p:
    return _lib.dsc_arange(ctx, n, dtype)


_lib.dsc_arange.argtypes = [_DscCtx, c_int, c_uint8]
_lib.dsc_arange.restype = _DscTensor_p


# extern dsc_tensor *dsc_randn(dsc_ctx *ctx,
#                              int n_dim,
#                              const int *shape,
#                              dsc_dtype dtype = DSC_DEFAULT_TYPE) noexcept;
def _dsc_randn(ctx: _DscCtx, shape: tuple[int, ...], dtype: c_uint8) -> _DscTensor_p:
    shape_type = c_int * len(shape)
    return _lib.dsc_randn(ctx, len(shape), shape_type(*shape), dtype)


_lib.dsc_randn.argtypes = [_DscCtx, c_int, POINTER(c_int), c_uint8]
_lib.dsc_randn.restype = _DscTensor_p


# extern dsc_tensor *dsc_add(dsc_ctx *ctx,
#                            dsc_tensor *__restrict xa,
#                            dsc_tensor *__restrict xb,
#                            dsc_tensor *__restrict out = nullptr) noexcept;
def _dsc_add(ctx: _DscCtx, xa: _DscTensor_p, xb: _DscTensor_p, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_add(ctx, xa, xb, out)


_lib.dsc_add.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, _DscTensor_p]
_lib.dsc_add.restype = _DscTensor_p


# extern dsc_tensor *dsc_sub(dsc_ctx *ctx,
#                            dsc_tensor *__restrict xa,
#                            dsc_tensor *__restrict xb,
#                            dsc_tensor *__restrict out = nullptr) noexcept;
def _dsc_sub(ctx: _DscCtx, xa: _DscTensor_p, xb: _DscTensor_p, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_sub(ctx, xa, xb, out)


_lib.dsc_sub.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, _DscTensor_p]
_lib.dsc_sub.restype = _DscTensor_p


# extern dsc_tensor *dsc_mul(dsc_ctx *ctx,
#                            dsc_tensor *__restrict xa,
#                            dsc_tensor *__restrict xb,
#                            dsc_tensor *__restrict out = nullptr) noexcept;
def _dsc_mul(ctx: _DscCtx, xa: _DscTensor_p, xb: _DscTensor_p, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_mul(ctx, xa, xb, out)


_lib.dsc_mul.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, _DscTensor_p]
_lib.dsc_mul.restype = _DscTensor_p


# extern dsc_tensor *dsc_div(dsc_ctx *ctx,
#                            dsc_tensor *__restrict xa,
#                            dsc_tensor *__restrict xb,
#                            dsc_tensor *__restrict out = nullptr) noexcept;
def _dsc_div(ctx: _DscCtx, xa: _DscTensor_p, xb: _DscTensor_p, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_div(ctx, xa, xb, out)


_lib.dsc_div.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, _DscTensor_p]
_lib.dsc_div.restype = _DscTensor_p


# extern dsc_tensor *dsc_pow(dsc_ctx *ctx,
#                            dsc_tensor *__restrict xa,
#                            dsc_tensor *__restrict xb,
#                            dsc_tensor *__restrict out = nullptr) noexcept;
def _dsc_pow(ctx: _DscCtx, xa: _DscTensor_p, xb: _DscTensor_p, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_pow(ctx, xa, xb, out)


_lib.dsc_pow.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, _DscTensor_p]
_lib.dsc_pow.restype = _DscTensor_p


def _dsc_addc_f32(ctx: _DscCtx, x: _DscTensor_p, val: float, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_addc_f32(ctx, x, c_float(val), out)


_lib.dsc_addc_f32.argtypes = [_DscCtx, _DscTensor_p, c_float, _DscTensor_p]
_lib.dsc_addc_f32.restype = _DscTensor_p


def _dsc_addc_f64(ctx: _DscCtx, x: _DscTensor_p, val: float, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_addc_f64(ctx, x, c_double(val), out)


_lib.dsc_addc_f64.argtypes = [_DscCtx, _DscTensor_p, c_double, _DscTensor_p]
_lib.dsc_addc_f64.restype = _DscTensor_p


def _dsc_addc_c32(ctx: _DscCtx, x: _DscTensor_p, val: complex, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_addc_c32(ctx, x, _C32(c_float(val.real), c_float(val.imag)), out)


_lib.dsc_addc_c32.argtypes = [_DscCtx, _DscTensor_p, _C32, _DscTensor_p]
_lib.dsc_addc_c32.restype = _DscTensor_p


def _dsc_addc_c64(ctx: _DscCtx, x: _DscTensor_p, val: complex, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_addc_c64(ctx, x, _C64(c_double(val.real), c_double(val.imag)), out)


_lib.dsc_addc_c64.argtypes = [_DscCtx, _DscTensor_p, _C64, _DscTensor_p]
_lib.dsc_addc_c64.restype = _DscTensor_p


def _dsc_subc_f32(ctx: _DscCtx, x: _DscTensor_p, val: float, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_subc_f32(ctx, x, c_float(val), out)


_lib.dsc_subc_f32.argtypes = [_DscCtx, _DscTensor_p, c_float, _DscTensor_p]
_lib.dsc_subc_f32.restype = _DscTensor_p


def _dsc_subc_f64(ctx: _DscCtx, x: _DscTensor_p, val: float, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_subc_f64(ctx, x, c_double(val), out)


_lib.dsc_subc_f64.argtypes = [_DscCtx, _DscTensor_p, c_double, _DscTensor_p]
_lib.dsc_subc_f64.restype = _DscTensor_p


def _dsc_subc_c32(ctx: _DscCtx, x: _DscTensor_p, val: complex, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_subc_c32(ctx, x, _C32(c_float(val.real), c_float(val.imag)), out)


_lib.dsc_subc_c32.argtypes = [_DscCtx, _DscTensor_p, _C32, _DscTensor_p]
_lib.dsc_subc_c32.restype = _DscTensor_p


def _dsc_subc_c64(ctx: _DscCtx, x: _DscTensor_p, val: complex, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_subc_c64(ctx, x, _C64(c_double(val.real), c_double(val.imag)), out)


_lib.dsc_subc_c64.argtypes = [_DscCtx, _DscTensor_p, _C64, _DscTensor_p]
_lib.dsc_subc_c64.restype = _DscTensor_p


def _dsc_mulc_f32(ctx: _DscCtx, x: _DscTensor_p, val: float, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_mulc_f32(ctx, x, c_float(val), out)


_lib.dsc_mulc_f32.argtypes = [_DscCtx, _DscTensor_p, c_float, _DscTensor_p]
_lib.dsc_mulc_f32.restype = _DscTensor_p


def _dsc_mulc_f64(ctx: _DscCtx, x: _DscTensor_p, val: float, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_mulc_f64(ctx, x, c_double(val), out)


_lib.dsc_mulc_f64.argtypes = [_DscCtx, _DscTensor_p, c_double, _DscTensor_p]
_lib.dsc_mulc_f64.restype = _DscTensor_p


def _dsc_mulc_c32(ctx: _DscCtx, x: _DscTensor_p, val: complex, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_mulc_c32(ctx, x, _C32(c_float(val.real), c_float(val.imag)), out)


_lib.dsc_mulc_c32.argtypes = [_DscCtx, _DscTensor_p, _C32, _DscTensor_p]
_lib.dsc_mulc_c32.restype = _DscTensor_p


def _dsc_mulc_c64(ctx: _DscCtx, x: _DscTensor_p, val: complex, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_mulc_c64(ctx, x, _C64(c_double(val.real), c_double(val.imag)), out)


_lib.dsc_mulc_c64.argtypes = [_DscCtx, _DscTensor_p, _C64, _DscTensor_p]
_lib.dsc_mulc_c64.restype = _DscTensor_p


def _dsc_divc_f32(ctx: _DscCtx, x: _DscTensor_p, val: float, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_divc_f32(ctx, x, c_float(val), out)


_lib.dsc_divc_f32.argtypes = [_DscCtx, _DscTensor_p, c_float, _DscTensor_p]
_lib.dsc_divc_f32.restype = _DscTensor_p


def _dsc_divc_f64(ctx: _DscCtx, x: _DscTensor_p, val: float, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_divc_f64(ctx, x, c_double(val), out)


_lib.dsc_divc_f64.argtypes = [_DscCtx, _DscTensor_p, c_double, _DscTensor_p]
_lib.dsc_divc_f64.restype = _DscTensor_p


def _dsc_divc_c32(ctx: _DscCtx, x: _DscTensor_p, val: complex, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_divc_c32(ctx, x, _C32(c_float(val.real), c_float(val.imag)), out)


_lib.dsc_divc_c32.argtypes = [_DscCtx, _DscTensor_p, _C32, _DscTensor_p]
_lib.dsc_divc_c32.restype = _DscTensor_p


def _dsc_divc_c64(ctx: _DscCtx, x: _DscTensor_p, val: complex, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_divc_c64(ctx, x, _C64(c_double(val.real), c_double(val.imag)), out)


_lib.dsc_divc_c64.argtypes = [_DscCtx, _DscTensor_p, _C64, _DscTensor_p]
_lib.dsc_divc_c64.restype = _DscTensor_p


def _dsc_powc_f32(ctx: _DscCtx, x: _DscTensor_p, val: float, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_powc_f32(ctx, x, c_float(val), out)


_lib.dsc_powc_f32.argtypes = [_DscCtx, _DscTensor_p, c_float, _DscTensor_p]
_lib.dsc_powc_f32.restype = _DscTensor_p


def _dsc_powc_f64(ctx: _DscCtx, x: _DscTensor_p, val: float, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_powc_f64(ctx, x, c_double(val), out)


_lib.dsc_powc_f64.argtypes = [_DscCtx, _DscTensor_p, c_double, _DscTensor_p]
_lib.dsc_powc_f64.restype = _DscTensor_p


def _dsc_powc_c32(ctx: _DscCtx, x: _DscTensor_p, val: complex, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_powc_c32(ctx, x, _C32(c_float(val.real), c_float(val.imag)), out)


_lib.dsc_powc_c32.argtypes = [_DscCtx, _DscTensor_p, _C32, _DscTensor_p]
_lib.dsc_powc_c32.restype = _DscTensor_p


def _dsc_powc_c64(ctx: _DscCtx, x: _DscTensor_p, val: complex, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_powc_c64(ctx, x, _C64(c_double(val.real), c_double(val.imag)), out)


_lib.dsc_powc_c64.argtypes = [_DscCtx, _DscTensor_p, _C64, _DscTensor_p]
_lib.dsc_powc_c64.restype = _DscTensor_p


#def _dsc_full_f32(ctx: _DscCtx, n: int, val: float) -> _DscTensor_p:
#    return _lib.dsc_full_f32(ctx, n, c_float(val))
#
#
#_lib.dsc_full_f32.argtypes = [_DscCtx, c_int, c_float]
#_lib.dsc_full_f32.restype = _DscTensor_p
#
#
#def _dsc_full_f64(ctx: _DscCtx, n: int, val: float) -> _DscTensor_p:
#    return _lib.dsc_full_f64(ctx, n, c_double(val))
#
#
#_lib.dsc_full_f64.argtypes = [_DscCtx, c_int, c_double]
#_lib.dsc_full_f64.restype = _DscTensor_p
#
#
#def _dsc_full_c32(ctx: _DscCtx, n: int, val: complex) -> _DscTensor_p:
#    return _lib.dsc_full_c32(ctx, n, c_float(val.real), c_float(val.imag))
#
#
#_lib.dsc_full_c32.argtypes = [_DscCtx, c_int, c_float, c_float]
#_lib.dsc_full_c32.restype = _DscTensor_p
#
#
#def _dsc_full_c64(ctx: _DscCtx, n: int, val: complex) -> _DscTensor_p:
#    return _lib.dsc_full_c64(ctx, n, c_double(val.real), c_double(val.imag))
#
#
#_lib.dsc_full_c64.argtypes = [_DscCtx, c_int, c_double, c_double]
#_lib.dsc_full_c64.restype = _DscTensor_p


# extern dsc_tensor *dsc_cast(dsc_ctx *ctx,
#                             dsc_tensor *__restrict x,
#                             dsc_dtype new_dtype) noexcept;
def _dsc_cast(ctx: _DscCtx, x: _DscTensor_p, dtype: c_uint8) -> _DscTensor_p:
    return _lib.dsc_cast(ctx, x, dtype)


_lib.dsc_cast.argtypes = [_DscCtx, _DscTensor_p, c_uint8]
_lib.dsc_cast.restype = _DscTensor_p


# extern dsc_tensor *dsc_cos(dsc_ctx *,
#                            const dsc_tensor *__restrict x,
#                            dsc_tensor *__restrict out) noexcept;
def _dsc_cos(ctx: _DscCtx, x: _DscTensor_p, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_cos(ctx, x, out)


_lib.dsc_cos.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p]
_lib.dsc_cos.restype = _DscTensor_p


# extern dsc_tensor *dsc_sin(dsc_ctx *,
#                            const dsc_tensor *__restrict x,
#                            dsc_tensor *__restrict out = nullptr) noexcept;
def _dsc_sin(ctx: _DscCtx, x: _DscTensor_p, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_sin(ctx, x, out)


_lib.dsc_sin.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p]
_lib.dsc_sin.restype = _DscTensor_p


# extern dsc_tensor *dsc_sinc(dsc_ctx *,
#                             const dsc_tensor *__restrict x,
#                             dsc_tensor *__restrict out = nullptr) noexcept;
def _dsc_sinc(ctx: _DscCtx, x: _DscTensor_p, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_sinc(ctx, x, out)


_lib.dsc_sinc.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p]
_lib.dsc_sinc.restype = _DscTensor_p


# extern dsc_tensor *dsc_logn(dsc_ctx *,
#                             const dsc_tensor *__restrict x,
#                             dsc_tensor *__restrict out = nullptr) noexcept;
def _dsc_logn(ctx: _DscCtx, x: _DscTensor_p, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_logn(ctx, x, out)


_lib.dsc_logn.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p]
_lib.dsc_logn.restype = _DscTensor_p


# extern dsc_tensor *dsc_log2(dsc_ctx *,
#                             const dsc_tensor *__restrict x,
#                             dsc_tensor *__restrict out = nullptr) noexcept;
def _dsc_log2(ctx: _DscCtx, x: _DscTensor_p, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_log2(ctx, x, out)


_lib.dsc_log2.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p]
_lib.dsc_log2.restype = _DscTensor_p


# extern dsc_tensor *dsc_log10(dsc_ctx *,
#                              const dsc_tensor *__restrict x,
#                              dsc_tensor *__restrict out = nullptr) noexcept;
def _dsc_log10(ctx: _DscCtx, x: _DscTensor_p, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_log10(ctx, x, out)


_lib.dsc_log10.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p]
_lib.dsc_log10.restype = _DscTensor_p


# extern dsc_tensor *dsc_exp(dsc_ctx *,
#                            const dsc_tensor *__restrict x,
#                            dsc_tensor *__restrict out = nullptr) noexcept;
def _dsc_exp(ctx: _DscCtx, x: _DscTensor_p, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_exp(ctx, x, out)


_lib.dsc_exp.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p]
_lib.dsc_exp.restype = _DscTensor_p


# extern dsc_tensor *dsc_sqrt(dsc_ctx *,
#                             const dsc_tensor *__restrict x,
#                             dsc_tensor *__restrict out = nullptr) noexcept;
def _dsc_sqrt(ctx: _DscCtx, x: _DscTensor_p, out: _DscTensor_p = None) -> _DscTensor_p:
    return _lib.dsc_sqrt(ctx, x, out)


_lib.dsc_sqrt.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p]
_lib.dsc_sqrt.restype = _DscTensor_p


# extern dsc_tensor *dsc_abs(dsc_ctx *,
#                             const dsc_tensor *__restrict x,
#                             dsc_tensor *__restrict out = nullptr) noexcept;
def _dsc_abs(ctx: _DscCtx, x: _DscTensor_p, out: _DscTensor_p = None) -> _DscTensor_p:
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


# extern dsc_tensor *dsc_sum(dsc_ctx *ctx,
#                            const dsc_tensor *DSC_RESTRICT x,
#                            dsc_tensor *DSC_RESTRICT out = nullptr,
#                            int axis = -1,
#                            bool keep_dims = true) noexcept;
def _dsc_sum(ctx: _DscCtx, x: _DscTensor_p, out: _DscTensor_p, axis: int = -1, keepdims: bool = True) -> _DscTensor_p:
    return _lib.dsc_sum(ctx, x, out, c_int(axis), c_bool(keepdims))


_lib.dsc_sum.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, c_int, c_bool]
_lib.dsc_sum.restype = _DscTensor_p


# extern dsc_tensor *dsc_mean(dsc_ctx *ctx,
#                            const dsc_tensor *DSC_RESTRICT x,
#                            dsc_tensor *DSC_RESTRICT out = nullptr,
#                            int axis = -1,
#                            bool keep_dims = true) noexcept;
def _dsc_mean(ctx: _DscCtx, x: _DscTensor_p, out: _DscTensor_p, axis: int = -1, keepdims: bool = True) -> _DscTensor_p:
    return _lib.dsc_mean(ctx, x, out, c_int(axis), c_bool(keepdims))


_lib.dsc_mean.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, c_int, c_bool]
_lib.dsc_mean.restype = _DscTensor_p


# extern dsc_tensor *dsc_fft(dsc_ctx *ctx,
#                            const dsc_tensor *DSC_RESTRICT x,
#                            dsc_tensor *DSC_RESTRICT out,
#                            int n = -1,
#                            int axis = -1) noexcept;
def _dsc_fft(ctx: _DscCtx, x: _DscTensor_p, out: _DscTensor_p = None, n: c_int = -1, axis: c_int = -1) -> _DscTensor_p:
    return _lib.dsc_fft(ctx, x, out, n, axis)


_lib.dsc_fft.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, c_int, c_int]
_lib.dsc_fft.restype = _DscTensor_p


# extern dsc_tensor *dsc_ifft(dsc_ctx *ctx,
#                             const dsc_tensor *DSC_RESTRICT x,
#                             dsc_tensor *DSC_RESTRICT out,
#                             int n = -1,
#                             int axis = -1) noexcept;
def _dsc_ifft(ctx: _DscCtx, x: _DscTensor_p, out: _DscTensor_p = None, n: c_int = -1, axis: c_int = -1) -> _DscTensor_p:
    return _lib.dsc_ifft(ctx, x, out, n, axis)


_lib.dsc_ifft.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, c_int, c_int]
_lib.dsc_ifft.restype = _DscTensor_p


# extern dsc_tensor *dsc_rfft(dsc_ctx *ctx,
#                             const dsc_tensor *DSC_RESTRICT x,
#                             dsc_tensor *DSC_RESTRICT out,
#                             int n = -1,
#                             int axis = -1) noexcept;
def _dsc_rfft(ctx: _DscCtx, x: _DscTensor_p, out: _DscTensor_p = None, n: c_int = -1, axis: c_int = -1) -> _DscTensor_p:
    return _lib.dsc_rfft(ctx, x, out, n, axis)


_lib.dsc_rfft.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, c_int, c_int]
_lib.dsc_rfft.restype = _DscTensor_p


# extern dsc_tensor *dsc_irfft(dsc_ctx *ctx,
#                              const dsc_tensor *DSC_RESTRICT x,
#                              dsc_tensor *DSC_RESTRICT out,
#                              int n = -1,
#                              int axis = -1) noexcept;
def _dsc_irfft(ctx: _DscCtx, x: _DscTensor_p, out: _DscTensor_p = None, n: c_int = -1, axis: c_int = -1) -> _DscTensor_p:
    return _lib.dsc_irfft(ctx, x, out, n, axis)


_lib.dsc_irfft.argtypes = [_DscCtx, _DscTensor_p, _DscTensor_p, c_int, c_int]
_lib.dsc_irfft.restype = _DscTensor_p


# extern dsc_tensor *dsc_fftfreq(dsc_ctx *ctx,
#                                int n,
#                                f64 d = 1.,
#                                dsc_dtype dtype = DSC_DEFAULT_TYPE) noexcept;
def _dsc_fftfreq(ctx: _DscCtx, n: c_int, d: c_double, dtype: c_uint8) -> _DscTensor_p:
    return _lib.dsc_fftfreq(ctx, n, d, dtype)


_lib.dsc_fftfreq.argtypes = [_DscCtx, c_int, c_double, c_uint8]
_lib.dsc_fftfreq.restype = _DscTensor_p


# extern dsc_tensor *dsc_rfftfreq(dsc_ctx *ctx,
#                                 int n,
#                                 f64 d = 1.,
#                                 dsc_dtype dtype = DSC_DEFAULT_TYPE) noexcept;
def _dsc_rfftfreq(ctx: _DscCtx, n: c_int, d: c_double, dtype: c_uint8) -> _DscTensor_p:
    return _lib.dsc_rfftfreq(ctx, n, d, dtype)


_lib.dsc_rfftfreq.argtypes = [_DscCtx, c_int, c_double, c_uint8]
_lib.dsc_rfftfreq.restype = _DscTensor_p
