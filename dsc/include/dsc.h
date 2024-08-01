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

#define DSC_UNUSED(x)        ((void) (x))
// Compute the next value of X aligned to Y
#define DSC_ALIGN(x, y)      (((x) + (y) - 1) & ~((y) - 1))
#define DSC_MAX(x, y)        ((x) > (y) ? (x) : (y))
#define DSC_MIN(x, y)        ((x) < (y) ? (x) : (y))
#define DSC_B_TO_KB(b)       ((f64)(b) / 1024.)
#define DSC_B_TO_MB(b)       ((f64)(b) / (1024. * 1024.))
#define DSC_MB(mb)           ((usize) ((mb) * 1024l * 1024l))
#define DSC_KB(kb)           ((usize) ((kb) * 1024l))

#if defined(__GNUC__)
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
#else
#   define DSC_STRICTLY_PURE
#   define DSC_PURE
#   define DSC_INLINE           inline
#   define DSC_NOINLINE
#endif

#define DSC_RESTRICT     __restrict
// Todo: use INFINITY macro for both types?
#define DSC_INF32        std::numeric_limits<f32>::max()
#define DSC_INF64        std::numeric_limits<f64>::max()

#if !defined(DSC_PAGE_SIZE)
#   define DSC_PAGE_SIZE ((usize) 4096)
#endif

#define DSC_MAX_DIMS     ((int) 4)

static_assert(DSC_MAX_DIMS == 4, "DSC_MAX_DIMS != 4 - update the code!");

#define CONST_FUNC_DECL(func, type) \
    extern dsc_tensor *dsc_##func##_##type(dsc_ctx *ctx,                        \
                                           dsc_tensor *DSC_RESTRICT x,          \
                                           type val,                            \
                                           dsc_tensor *DSC_RESTRICT out = nullptr) noexcept;

#define DSC_TENSOR_DIMS_0(PTR)   const int d0_##PTR = (PTR)->shape[0]
#define DSC_TENSOR_DIMS_1(PTR)   DSC_TENSOR_DIMS_0(PTR); \
                                 const int d1_##PTR = (PTR)->shape[1]
#define DSC_TENSOR_DIMS_2(PTR)   DSC_TENSOR_DIMS_1(PTR); \
                                 const int d2_##PTR = (PTR)->shape[2]
#define DSC_TENSOR_DIMS_3(PTR)   DSC_TENSOR_DIMS_2(PTR); \
                                 const int d3_##PTR = (PTR)->shape[3]

#define DSC_TENSOR_STRIDES_0(PTR)    const int d0_stride_##PTR = (PTR)->stride[0]
#define DSC_TENSOR_STRIDES_1(PTR)    DSC_TENSOR_STRIDES_0(PTR); \
                                     const int d1_stride_##PTR = (PTR)->stride[1]
#define DSC_TENSOR_STRIDES_2(PTR)    DSC_TENSOR_STRIDES_1(PTR); \
                                     const int d2_stride_##PTR = (PTR)->stride[2]
#define DSC_TENSOR_STRIDES_3(PTR)    DSC_TENSOR_STRIDES_2(PTR); \
                                     const int d3_stride_##PTR = (PTR)->stride[3]

#define DSC_TENSOR_DIMS(PTR, n)         DSC_TENSOR_DIMS_##n(PTR)
#define DSC_TENSOR_STRIDES(PTR, n)      DSC_TENSOR_STRIDES_##n(PTR)
#define DSC_TENSOR_FIELDS(PTR, n)       DSC_TENSOR_DIMS(PTR, n); DSC_TENSOR_STRIDES(PTR, n)
#define DSC_TENSOR_DATA(T, PTR)         T *DSC_RESTRICT PTR##_data = (T *) (PTR)->data

// Todo: upper or lower case?
#define dsc_offset_0(PTR)    ((d0) * (d0_stride_##PTR))
#define dsc_offset_1(PTR)    ((dsc_offset_0(PTR)) + ((d1) * (d1_stride_##PTR)))
#define dsc_offset_2(PTR)    ((dsc_offset_1(PTR)) + ((d2) * (d2_stride_##PTR)))
#define dsc_offset_3(PTR)    ((dsc_offset_2(PTR)) + ((d3) * (d3_stride_##PTR)))
#define dsc_offset(PTR, n)   dsc_offset_##n(PTR)

#define dsc_broadcast_offset_0(PTR)  (((d0) % (d0_##PTR)) * (d0_stride_##PTR))
#define dsc_broadcast_offset_1(PTR)  ((dsc_broadcast_offset_0(PTR)) + (((d1) % (d1_##PTR)) * (d1_stride_##PTR)))
#define dsc_broadcast_offset_2(PTR)  ((dsc_broadcast_offset_1(PTR)) + (((d2) % (d2_##PTR)) * (d2_stride_##PTR)))
#define dsc_broadcast_offset_3(PTR)  ((dsc_broadcast_offset_2(PTR)) + (((d3) % (d3_##PTR)) * (d3_stride_##PTR)))
#define dsc_broadcast_offset(PTR, n) dsc_broadcast_offset_##n(PTR)

#define dsc_new_like(CTX, PTR) (dsc_new_tensor((CTX), (PTR)->n_dim, (PTR)->shape, (PTR)->dtype))

#define dsc_for_0(PTR)   for (int d0 = 0; d0 < (d0_##PTR); ++d0)
#define dsc_for_1(PTR)   dsc_for_0(PTR) \
                         for (int d1 = 0; d1 < (d1_##PTR); ++d1)
#define dsc_for_2(PTR)   dsc_for_1(PTR) \
                         for (int d2 = 0; d2 < (d2_##PTR); ++d2)
#define dsc_for_3(PTR)   dsc_for_2(PTR) \
                         for (int d3 = 0; d3 < (d3_##PTR); ++d3)
#define dsc_for(PTR, n)  dsc_for_##n(PTR)

#define dsc_tensor_dim(PTR, dim) (((dim) < 0) ? (DSC_MAX_DIMS + (dim)) : (DSC_MAX_DIMS - (PTR)->n_dim + (dim)))

#if defined(__cplusplus)
extern "C" {
#endif

struct dsc_ctx;
struct dsc_obj;
struct dsc_fft_plan;

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

extern dsc_ctx *dsc_ctx_init(usize nb) noexcept;

extern dsc_fft_plan *dsc_plan_fft(dsc_ctx *ctx, int n,
                                  dsc_dtype dtype = dsc_dtype::F64) noexcept;

extern void dsc_ctx_free(dsc_ctx *ctx) noexcept;

extern void dsc_ctx_clear(dsc_ctx *ctx) noexcept;

extern dsc_tensor *dsc_new_tensor(dsc_ctx *ctx,
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

// Todo: if we decide to stick with this pattern for external function (c-like api + internal generic impl with templates)
//  then it makes sense to use a more generic macro, similar to CONST_FUNC_DECL that can take care also of the arguments to declare all the functions.
extern dsc_tensor *dsc_log_space_f32(dsc_ctx *ctx,
                                     f32 start,
                                     f32 stop,
                                     int n,
                                     f32 base = 10.f) noexcept;

extern dsc_tensor *dsc_log_space_f64(dsc_ctx *ctx,
                                     f64 start,
                                     f64 stop,
                                     int n,
                                     f64 base = 10.) noexcept;

extern dsc_tensor *dsc_interp1d_f32(dsc_ctx *ctx,
                                    const dsc_tensor *x,
                                    const dsc_tensor *y,
                                    const dsc_tensor *xp,
                                    f32 left = DSC_INF32,
                                    f32 right = DSC_INF32) noexcept;

extern dsc_tensor *dsc_interp1d_f64(dsc_ctx *ctx,
                                    const dsc_tensor *x,
                                    const dsc_tensor *y,
                                    const dsc_tensor *xp,
                                    f64 left = DSC_INF64,
                                    f64 right = DSC_INF64) noexcept;

extern dsc_tensor *dsc_cast(dsc_ctx *ctx,
                            dsc_tensor *DSC_RESTRICT x,
                            dsc_dtype new_dtype) noexcept;

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

extern dsc_tensor *dsc_cos(dsc_ctx *ctx,
                           const dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out = nullptr) noexcept;

extern dsc_tensor *dsc_sin(dsc_ctx *ctx,
                           const dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out = nullptr) noexcept;

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

#if defined(__cplusplus)
}
#endif
