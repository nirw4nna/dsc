// Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "dsc_dtype.h"

#define DSC_LOG_FATAL(format, ...)          \
    do {                                    \
        DSC_LOG_ERR(format, ##__VA_ARGS__); \
        exit(EXIT_FAILURE);                 \
    } while(0)
#define DSC_LOG_ERR(format, ...)   fprintf(stderr, "%s: " format"\n",__func__, ##__VA_ARGS__)
#define DSC_LOG_INFO(format, ...)  fprintf(stdout, "%s: " format"\n",__func__, ##__VA_ARGS__)

#define DSC_ASSERT(x)                                                           \
    do {                                                                        \
        if (!(x)) {                                                             \
            fprintf(stderr, "DSC_ASSERT: %s:%d %s\n", __FILE__, __LINE__, #x);  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while(0)

#if defined(DSC_DEBUG)
#   define DSC_LOG_DEBUG(format, ...)  DSC_LOG_INFO(format, ##__VA_ARGS__)
#else
#   define DSC_LOG_DEBUG(format, ...)  ((void) 0)
#endif

#define DSC_INVALID_CASE(format, ...)   \
    default:                            \
        DSC_LOG_FATAL(format, ##__VA_ARGS__)

#define DSC_UNUSED(x)        ((void) (x))
// Compute the next value of X aligned to Y
#define DSC_ALIGN(x, y)      (((x) + (y) - 1) & ~((y) - 1))
#define DSC_MAX(x, y)        ((x) > (y) ? (x) : (y))
#define DSC_MIN(x, y)        ((x) < (y) ? (x) : (y))
#define DSC_B_TO_KB(b)       ((f64)(b) / 1024.)
#define DSC_B_TO_MB(b)       ((f64)(b) / (1024. * 1024.))
#define DSC_MB(mb)           ((usize) ((mb) * 1024l * 1024l))
#define DSC_KB(kb)           ((usize) ((kb) * 1024l))

#if defined(__GNUC__) && !defined(EMSCRIPTEN)
// A 'strictly pure' function is a function whose return value doesn't depend on the global state of the program,
// this means that it must not access global variables subject to change or access parameters passed by pointer
// unless the actual value of the pointer does not change after the first invocation.
// A 'pure' function is basically the same thing without the restriction on global state change, this means
// that a 'pure' function can take in and read the value of parameters passed by pointer even if that value
// changes between subsequent invocations.
#   define DSC_STRICTLY_PURE    __attribute_const__
#   define DSC_PURE             __attribute_pure__
#   define DSC_INLINE           inline __attribute__((always_inline))
#   define DSC_NOINLINE         __attribute__((noinline))
#   define DSC_MALLOC           __attribute__((malloc))
#else
#   define DSC_STRICTLY_PURE
#   define DSC_PURE
#   define DSC_INLINE           inline
#   define DSC_NOINLINE
#   define DSC_MALLOC
#endif

#define DSC_RESTRICT    __restrict

#if !defined(DSC_MAX_DIMS)
#   define DSC_MAX_DIMS ((int) 4)
#endif

static_assert(DSC_MAX_DIMS == 4, "DSC_MAX_DIMS != 4 - update the code");

#define CONST_FUNC_DECL(func, type) \
    extern dsc_tensor *dsc_##func##_##type(dsc_ctx *ctx,    \
                                           dsc_tensor *x,   \
                                           type val,        \
                                           dsc_tensor *out = nullptr) noexcept;

#define DSC_TENSOR_DATA(T, PTR) T *DSC_RESTRICT PTR##_data = (T *) (PTR)->data

#define dsc_new_like(CTX, PTR)      (dsc_new_tensor((CTX), (PTR)->n_dim, &(PTR)->shape[DSC_MAX_DIMS - (PTR)->n_dim], (PTR)->dtype))
#define dsc_tensor_dim(PTR, dim)    (((dim) < 0) ? (DSC_MAX_DIMS + (dim)) : (DSC_MAX_DIMS - (PTR)->n_dim + (dim)))

#if defined(__cplusplus)
extern "C" {
#endif

struct dsc_ctx;
struct dsc_obj;
struct dsc_fft_plan;
enum dsc_fft_type : u8;

struct dsc_tensor {
    // The shape of this tensor, right-aligned. For example a 1D tensor T of 4 elements
    // will have dim = [1, 1, 1, 4].
    int shape[DSC_MAX_DIMS];
    // Stride for a given dimension expressed in number of bytes.
    int stride[DSC_MAX_DIMS];
    void *data;
    int ne;
    int n_dim;
    dsc_dtype dtype;
};

static DSC_INLINE f64 dsc_timer() noexcept {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (f64) ts.tv_sec + (f64) ts.tv_nsec * 1.e-9;
}

// ============================================================
// Initialization

extern dsc_ctx *dsc_ctx_init(usize nb) noexcept;

extern dsc_fft_plan *dsc_plan_fft(dsc_ctx *ctx, int n,
                                  dsc_fft_type fft_type,
                                  dsc_dtype dtype = dsc_dtype::F64) noexcept;

// ============================================================
// Cleanup/Teardown

extern void dsc_ctx_free(dsc_ctx *ctx) noexcept;

extern void dsc_ctx_clear(dsc_ctx *ctx) noexcept;

// ============================================================
// Tensor Creation

extern DSC_MALLOC dsc_tensor *dsc_new_tensor(dsc_ctx *ctx,
                                             int n_dim,
                                             const int *shape,
                                             dsc_dtype dtype) noexcept;

extern dsc_tensor *dsc_tensor_1d(dsc_ctx *ctx,
                                 dsc_dtype dtype,
                                 int dim1) noexcept;

extern dsc_tensor *dsc_tensor_2d(dsc_ctx *ctx,
                                 dsc_dtype dtype,
                                 int dim1, int dim2) noexcept;

extern dsc_tensor *dsc_tensor_3d(dsc_ctx *ctx,
                                 dsc_dtype dtype,
                                 int dim1, int dim2,
                                 int dim3) noexcept;

extern dsc_tensor *dsc_tensor_4d(dsc_ctx *ctx,
                                 dsc_dtype dtype,
                                 int dim1, int dim2,
                                 int dim3, int dim4) noexcept;

extern dsc_tensor *dsc_arange(dsc_ctx *ctx,
                              int n,
                              dsc_dtype dtype = DSC_DEFAULT_TYPE) noexcept;

extern dsc_tensor *dsc_randn(dsc_ctx *ctx,
                             int n_dim,
                             const int *shape,
                             dsc_dtype dtype = DSC_DEFAULT_TYPE) noexcept;

extern dsc_tensor *dsc_cast(dsc_ctx *ctx,
                            dsc_tensor *DSC_RESTRICT x,
                            dsc_dtype new_dtype) noexcept;

// ============================================================
// Binary Operations (Vector)

extern dsc_tensor *dsc_add(dsc_ctx *ctx,
                           dsc_tensor *DSC_RESTRICT xa,
                           dsc_tensor *DSC_RESTRICT xb,
                           dsc_tensor *DSC_RESTRICT out = nullptr) noexcept;

extern dsc_tensor *dsc_sub(dsc_ctx *ctx,
                           dsc_tensor *DSC_RESTRICT xa,
                           dsc_tensor *DSC_RESTRICT xb,
                           dsc_tensor *DSC_RESTRICT out = nullptr) noexcept;

extern dsc_tensor *dsc_mul(dsc_ctx *ctx,
                           dsc_tensor *DSC_RESTRICT xa,
                           dsc_tensor *DSC_RESTRICT xb,
                           dsc_tensor *DSC_RESTRICT out = nullptr) noexcept;

extern dsc_tensor *dsc_div(dsc_ctx *ctx,
                           dsc_tensor *DSC_RESTRICT xa,
                           dsc_tensor *DSC_RESTRICT xb,
                           dsc_tensor *DSC_RESTRICT out = nullptr) noexcept;

extern dsc_tensor *dsc_pow(dsc_ctx *ctx,
                           dsc_tensor *DSC_RESTRICT xa,
                           dsc_tensor *DSC_RESTRICT xb,
                           dsc_tensor *DSC_RESTRICT out = nullptr) noexcept;

// ============================================================
// Binary Operations (Scalar)

CONST_FUNC_DECL(addc, f32)
CONST_FUNC_DECL(addc, f64)
CONST_FUNC_DECL(addc, c32)
CONST_FUNC_DECL(addc, c64)

CONST_FUNC_DECL(subc, f32)
CONST_FUNC_DECL(subc, f64)
CONST_FUNC_DECL(subc, c32)
CONST_FUNC_DECL(subc, c64)

CONST_FUNC_DECL(mulc, f32)
CONST_FUNC_DECL(mulc, f64)
CONST_FUNC_DECL(mulc, c32)
CONST_FUNC_DECL(mulc, c64)

CONST_FUNC_DECL(divc, f32)
CONST_FUNC_DECL(divc, f64)
CONST_FUNC_DECL(divc, c32)
CONST_FUNC_DECL(divc, c64)

CONST_FUNC_DECL(powc, f32)
CONST_FUNC_DECL(powc, f64)
CONST_FUNC_DECL(powc, c32)
CONST_FUNC_DECL(powc, c64)

// ============================================================
// Unary Operations

extern dsc_tensor *dsc_cos(dsc_ctx *ctx,
                           const dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out = nullptr) noexcept;

extern dsc_tensor *dsc_sin(dsc_ctx *ctx,
                           const dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out = nullptr) noexcept;

extern dsc_tensor *dsc_sinc(dsc_ctx *ctx,
                            const dsc_tensor *DSC_RESTRICT x,
                            dsc_tensor *DSC_RESTRICT out = nullptr) noexcept;

extern dsc_tensor *dsc_logn(dsc_ctx *ctx,
                            const dsc_tensor *DSC_RESTRICT x,
                            dsc_tensor *DSC_RESTRICT out = nullptr) noexcept;

extern dsc_tensor *dsc_log2(dsc_ctx *ctx,
                            const dsc_tensor *DSC_RESTRICT x,
                            dsc_tensor *DSC_RESTRICT out = nullptr) noexcept;

extern dsc_tensor *dsc_log10(dsc_ctx *ctx,
                             const dsc_tensor *DSC_RESTRICT x,
                             dsc_tensor *DSC_RESTRICT out = nullptr) noexcept;

extern dsc_tensor *dsc_exp(dsc_ctx *ctx,
                           const dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out = nullptr) noexcept;

extern dsc_tensor *dsc_sqrt(dsc_ctx *ctx,
                            const dsc_tensor *DSC_RESTRICT x,
                            dsc_tensor *DSC_RESTRICT out = nullptr) noexcept;

extern dsc_tensor *dsc_abs(dsc_ctx *ctx,
                           const dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out = nullptr) noexcept;

extern dsc_tensor *dsc_angle(dsc_ctx *ctx,
                             const dsc_tensor *DSC_RESTRICT x) noexcept;

// conj and real are NOP if the input is real meaning x will be returned as is.
extern dsc_tensor *dsc_conj(dsc_ctx *ctx,
                            dsc_tensor *DSC_RESTRICT x) noexcept;

extern dsc_tensor *dsc_real(dsc_ctx *ctx,
                            dsc_tensor *DSC_RESTRICT x) noexcept;

extern dsc_tensor *dsc_imag(dsc_ctx *ctx,
                            const dsc_tensor *DSC_RESTRICT x) noexcept;

// Modified Bessel function of the first kind order 0
extern dsc_tensor *dsc_i0(dsc_ctx *ctx,
                          const dsc_tensor *DSC_RESTRICT x) noexcept;

// ============================================================
// Unary Operations Along Axis

extern dsc_tensor *dsc_sum(dsc_ctx *ctx,
                           const dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out = nullptr,
                           int axis = -1,
                           bool keep_dims = true) noexcept;

extern dsc_tensor *dsc_mean(dsc_ctx *ctx,
                            const dsc_tensor *DSC_RESTRICT x,
                            dsc_tensor *DSC_RESTRICT out = nullptr,
                            int axis = -1,
                            bool keep_dims = true) noexcept;

extern dsc_tensor *dsc_max(dsc_ctx *ctx,
                           const dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out = nullptr,
                           int axis = -1,
                           bool keep_dims = true) noexcept;

// ============================================================
// Fourier Transforms
//
// FFTs are always performed out-of-place. If the out param is provided then
// it will be used to store the result otherwise a new tensor will be allocated.
// The axis parameter specifies over which dimension the FFT must be performed,
// if x is 1-dimensional it will be ignored.
// If n is not specified then the FFT will be assumed to have the same size as the
// selected dimension of x otherwise that dimension will be padded/cropped to the value of n
// before performing the FFT.
extern dsc_tensor *dsc_fft(dsc_ctx *ctx,
                           const dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out = nullptr,
                           int n = -1,
                           int axis = -1) noexcept;

extern dsc_tensor *dsc_ifft(dsc_ctx *ctx,
                            const dsc_tensor *DSC_RESTRICT x,
                            dsc_tensor *DSC_RESTRICT out = nullptr,
                            int n = -1,
                            int axis = -1) noexcept;

extern dsc_tensor *dsc_rfft(dsc_ctx *ctx,
                            const dsc_tensor *DSC_RESTRICT x,
                            dsc_tensor *DSC_RESTRICT out = nullptr,
                            int n = -1,
                            int axis = -1) noexcept;

extern dsc_tensor *dsc_irfft(dsc_ctx *ctx,
                             const dsc_tensor *DSC_RESTRICT x,
                             dsc_tensor *DSC_RESTRICT out = nullptr,
                             int n = -1,
                             int axis = -1) noexcept;

extern dsc_tensor *dsc_fftfreq(dsc_ctx *ctx,
                               int n,
                               f64 d = 1.,
                               dsc_dtype dtype = DSC_DEFAULT_TYPE) noexcept;

extern dsc_tensor *dsc_rfftfreq(dsc_ctx *ctx,
                                int n,
                                f64 d = 1.,
                                dsc_dtype dtype = DSC_DEFAULT_TYPE) noexcept;

#if defined(__cplusplus)
}
#endif
