// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

// =============================================================== //
// =========================== Notepad =========================== //
// =============================================================== //
// (1) It's probably a good idea to add a struct for the arguments //
//     of dsc_new_tensor, mixing new params with defaults can lead //
//     to nasty bugs                                               //
// (2) Create a macro to validate tensors? It's probably a good    //
//     idea to always check by defaults both tensor != nullptr and //
//     tensor->buf != nullptr                                      //
// (3) Sometimes the context in Python is freed before all the     //
//     associated tensors are freed. This will SEGFAULT! It makes  //
//     sense to just not free the context in Python for now        //
// (4) Evaluate the iterator approach (check codegen with godbolt) //
// =============================================================== //

#include <cstdio>
#include <cstdlib>
#include "dsc_dtype.h"


#if !defined(DSC_MAX_OBJS)
#    define DSC_MAX_OBJS     ((int) 1'000)
#endif
#define DSC_MAX_DEVICES      ((int) 1)
#define DSC_DEFAULT_DEVICE   CPU
#define DSC_COMPARISON_OPS   ((int) 6)

static_assert(DSC_MAX_DEVICES == 1, "DSC_MAX_DEVICES != 1 - update the code");
static_assert(DSC_COMPARISON_OPS == 6, "DSC_COMPARISON_OPS != 6 - update the code");

#define DSC_ASSERT(x)                                                           \
    do {                                                                        \
        if (!(x)) {                                                             \
            fprintf(stderr, "DSC_ASSERT: %s:%d %s\n", __FILE__, __LINE__, #x);  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while(0)

#define DSC_LOG_FATAL(format, ...)                                            \
    do {                                                                      \
        fprintf(stderr, "[FATAL] %s: " format "\n", __func__, ##__VA_ARGS__); \
        exit(EXIT_FAILURE);                                                   \
    } while (0)

#if DSC_LOG_LEVEL >= 3
#    define DSC_LOG_DEBUG(format, ...)  ((void) 0)
#    define DSC_LOG_INFO(format, ...)   ((void) 0)
#    define DSC_LOG_ERR(format, ...)    ((void) 0)
#elif DSC_LOG_LEVEL >= 2
#    define DSC_LOG_DEBUG(format, ...)  ((void) 0)
#    define DSC_LOG_INFO(format, ...)   ((void) 0)
#    define DSC_LOG_ERR(format, ...)    fprintf(stderr, "[ERROR] %s: " format"\n",__func__, ##__VA_ARGS__)
#elif DSC_LOG_LEVEL >= 1
#    define DSC_LOG_DEBUG(format, ...)  ((void) 0)
#    define DSC_LOG_INFO(format, ...)   fprintf(stdout, "[INFO ] %s: " format"\n",__func__, ##__VA_ARGS__)
#    define DSC_LOG_ERR(format, ...)    fprintf(stderr, "[ERROR] %s: " format"\n",__func__, ##__VA_ARGS__)
#else
#    define DSC_LOG_DEBUG(format, ...)  fprintf(stdout, "[DEBUG] %s: " format"\n",__func__, ##__VA_ARGS__)
#    define DSC_LOG_INFO(format, ...)   fprintf(stdout, "[INFO ] %s: " format"\n",__func__, ##__VA_ARGS__)
#    define DSC_LOG_ERR(format, ...)    fprintf(stderr, "[ERROR] %s: " format"\n",__func__, ##__VA_ARGS__)
#endif

#define DSC_INVALID_CASE(format, ...)   \
    default:                            \
        DSC_LOG_FATAL(format, ##__VA_ARGS__)

#define DSC_UNUSED(x)        ((void) (x))
// Compute the next value of X aligned to Y
#define DSC_ALIGN(x, y)      (((x) + (y) - 1) & ~((y) - 1))
#define DSC_MAX(x, y)        ((x) > (y) ? (x) : (y))
#define DSC_MIN(x, y)        ((x) < (y) ? (x) : (y))
#define DSC_CEIL(x, y)       (((x) + ((y) - 1)) / (y))
#define DSC_B_TO_KB(b)       ((f64) (b) / 1024.)
#define DSC_B_TO_MB(b)       ((f64) (b) / (1024. * 1024.))
#define DSC_MB(mb)           ((usize) ((mb) * 1024ULL * 1024ULL))
#define DSC_KB(kb)           ((usize) ((kb) * 1024ULL))

// A 'strictly pure' function is a function whose return value doesn't depend on the global state of the program,
// this means that it must not access global variables subject to change or access parameters passed by pointer
// unless the actual value of the pointer does not change after the first invocation.
// A 'pure' function is basically the same thing without the restriction on global state change, this means
// that a 'pure' function can take in and read the value of parameters passed by pointer even if that value
// changes between subsequent invocations.
#if defined(__GNUC__)
#    define DSC_INLINE          inline __attribute__((always_inline))
#    define DSC_STRICTLY_PURE   __attribute__((const))
#    define DSC_PURE            __attribute__((pure))
#else
#    define DSC_INLINE          inline
#    define DSC_STRICTLY_PURE
#    define DSC_PURE
#endif

#define DSC_RESTRICT __restrict

#if !defined(DSC_MAX_DIMS)
#    define DSC_MAX_DIMS ((int) 4)
#endif

static_assert(DSC_MAX_DIMS == 4, "DSC_MAX_DIMS != 4 - update the code");

#define DSC_VALUE_NONE          INT32_MAX
#define DSC_DATA_ALIAS(T, X)    T *X##_data = (T *) (X)->buf->data
#define DSC_DATA(T, X)          T *DSC_RESTRICT X##_data = (T *) (X)->buf->data

#define dsc_tensor_dim_idx(X, dim)    (((dim) < 0) ? (DSC_MAX_DIMS + (dim)) : (DSC_MAX_DIMS - (X)->n_dim + (dim)))
// Note: dsc_tensor_get_dim() MUST NOT be used with the result of dsc_tensor_dim_idx()!
#define dsc_tensor_get_dim(X, dim)    ((X)->shape[dsc_tensor_dim_idx((X), (dim))])
#define dsc_tensor_get_stride(X, dim) ((X)->stride[dsc_tensor_dim_idx((X), (dim))])
#define dsc_new_like(CTX, X)          (dsc_new_tensor((CTX), (X)->n_dim, &dsc_tensor_get_dim(X, 0), (X)->dtype, (X)->device))
#define dsc_new_view(CTX, X)          (dsc_new_tensor((CTX), (X)->n_dim, &dsc_tensor_get_dim(X, 0), (X)->dtype, (X)->device, (X)->buf))
#define dsc_for(idx, X)               for (int idx = 0; idx < (X)->ne; ++idx)
#define dsc_is_scalar(X)              (X)->ne == 1

#if defined(__cplusplus)
extern "C" {
#endif

struct dsc_ctx;
struct dsc_data_buffer;
struct dsc_trace;
enum dsc_trace_phase : char;

struct dsc_traces {
    dsc_trace *traces;
    u64 n_traces;
};

enum dsc_device_type : i8 {
    DEFAULT = -1,
    CPU,
};

static constexpr const char *DSC_DEVICE_NAMES[DSC_MAX_DEVICES] = {
        "CPU",
};

enum dsc_comparison_op : u8 {
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE
};

struct dsc_tensor {
    // The shape of this tensor, right-aligned. For example a 1D tensor T of 4 elements
    // will have dim = [1, 1, 1, 4].
    int shape[DSC_MAX_DIMS];
    // Stride for a given dimension expressed in number of elements.
    int stride[DSC_MAX_DIMS];
    dsc_data_buffer *buf;
    int ne;
    int n_dim;
    dsc_dtype dtype;
    dsc_device_type device;
};

struct dsc_pair {
    dsc_tensor *first, *second;
};

struct dsc_slice {
    union {
        int d[3];
        struct {
            int start, stop, step;
        };
    };
};

// ============================================================
// Initialization

extern dsc_ctx *dsc_ctx_init(usize mem_size);

// ============================================================
// Cleanup/Teardown

extern void dsc_ctx_free(dsc_ctx *ctx);

extern void dsc_tensor_free(dsc_ctx *ctx, dsc_tensor *x);

// ============================================================
// Utilities

extern usize dsc_used_mem(dsc_ctx *ctx);

extern void dsc_print_mem_usage(dsc_ctx *ctx);

// ============================================================
// Tracing

extern bool dsc_tracing_enabled(dsc_ctx *);

extern void dsc_traces_record(dsc_ctx *ctx,
                              bool record = true);

extern void dsc_insert_trace(dsc_ctx *ctx,
                             const char *name,
                             const char *cat,
                             u64 ts,
                             dsc_trace_phase phase);

extern void dsc_dump_traces(dsc_ctx *ctx);

// ============================================================
// Tensor Creation

extern void dsc_tensor_set_buffer(dsc_ctx *,
                                  dsc_tensor *DSC_RESTRICT x,
                                  dsc_data_buffer *buf);

// TODO: (1) (2)
extern dsc_tensor *dsc_new_tensor(dsc_ctx *ctx,
                                  int n_dim,
                                  const int *shape,
                                  dsc_dtype dtype,
                                  dsc_device_type device = DEFAULT,
                                  dsc_data_buffer *buf = nullptr,
                                  bool lazy = false,
                                  const void *DSC_RESTRICT data = nullptr,
                                  dsc_device_type data_device = DEFAULT);

extern dsc_tensor *dsc_view(dsc_ctx *ctx,
                            const dsc_tensor *x);

extern dsc_tensor *dsc_tensor_1d(dsc_ctx *ctx,
                                 dsc_dtype dtype,
                                 int dim1,
                                 dsc_device_type device = DEFAULT,
                                 const void *DSC_RESTRICT data = nullptr,
                                 dsc_device_type data_device = DEFAULT);

extern dsc_tensor *dsc_tensor_2d(dsc_ctx *ctx,
                                 dsc_dtype dtype,
                                 int dim1, int dim2,
                                 dsc_device_type device = DEFAULT,
                                 const void *DSC_RESTRICT data = nullptr,
                                 dsc_device_type data_device = DEFAULT);

extern dsc_tensor *dsc_tensor_3d(dsc_ctx *ctx,
                                 dsc_dtype dtype,
                                 int dim1, int dim2,
                                 int dim3,
                                 dsc_device_type device = DEFAULT,
                                 const void *DSC_RESTRICT data = nullptr,
                                 dsc_device_type data_device = DEFAULT);

extern dsc_tensor *dsc_tensor_4d(dsc_ctx *ctx,
                                 dsc_dtype dtype,
                                 int dim1, int dim2,
                                 int dim3, int dim4,
                                 dsc_device_type device = DEFAULT,
                                 const void *DSC_RESTRICT data = nullptr,
                                 dsc_device_type data_device = DEFAULT);

extern dsc_tensor *dsc_wrap_bool(dsc_ctx *ctx,
                                 bool val,
                                 dsc_device_type device = DEFAULT);

extern dsc_tensor *dsc_wrap_i32(dsc_ctx *ctx,
                                i32 val,
                                dsc_device_type device = DEFAULT);

extern dsc_tensor *dsc_wrap_f32(dsc_ctx *ctx,
                                f32 val,
                                dsc_device_type device = DEFAULT);

extern dsc_tensor *dsc_wrap_f64(dsc_ctx *ctx,
                                f64 val,
                                dsc_device_type device = DEFAULT);

extern dsc_tensor *dsc_arange(dsc_ctx *ctx,
                              int n,
                              dsc_dtype dtype = I32,
                              dsc_device_type device = DEFAULT);

extern dsc_tensor *dsc_randn(dsc_ctx *ctx,
                             int n_dim,
                             const int *shape,
                             dsc_dtype dtype = DSC_DEFAULT_TYPE,
                             dsc_device_type device = DEFAULT);

extern dsc_pair dsc_topk(dsc_ctx *ctx,
                         const dsc_tensor *DSC_RESTRICT x,
                         int k,
                         int axis = -1,
                         bool largest = true);

extern dsc_tensor *dsc_multinomial(dsc_ctx *ctx,
                                   const dsc_tensor *DSC_RESTRICT x,
                                   int num_samples);

extern dsc_tensor *dsc_cast(dsc_ctx *ctx,
                            dsc_tensor *DSC_RESTRICT x,
                            dsc_dtype new_dtype);

// ============================================================
// Tensor Manipulation

extern void dsc_copy(dsc_ctx *ctx,
                     dsc_tensor *DSC_RESTRICT x,
                     void *DSC_RESTRICT data,
                     usize nb,
                     dsc_device_type data_device = DEFAULT);

extern dsc_tensor *dsc_reshape(dsc_ctx *ctx,
                               const dsc_tensor *DSC_RESTRICT x,
                               int dimensions...);

extern dsc_tensor *dsc_concat(dsc_ctx *ctx,
                              int axis,
                              int tensors...);

extern dsc_tensor *dsc_transpose(dsc_ctx *ctx,
                                 const dsc_tensor *DSC_RESTRICT x,
                                 int axes...);

extern dsc_tensor *dsc_tril(dsc_ctx *ctx,
                            const dsc_tensor *DSC_RESTRICT x,
                            int diagonal = 0,
                            dsc_tensor *DSC_RESTRICT out = nullptr);

// ============================================================
// Indexing and Slicing
//
// All indexing and slicing operations will return a new tensor.
// If the number of indexes passed to dsc_tensor_get_idx is equal to the number of
// dimensions of x then a new tensor will be allocated with a single element,
// the caller must take care of unwrapping it if needed.
extern dsc_tensor *dsc_tensor_get_idx(dsc_ctx *ctx,
                                      const dsc_tensor *DSC_RESTRICT x,
                                      int indexes...);

extern dsc_tensor *dsc_tensor_get_slice(dsc_ctx *ctx,
                                        const dsc_tensor *DSC_RESTRICT x,
                                        int slices...);

extern dsc_tensor *dsc_tensor_get_tensor(dsc_ctx *ctx,
                                         const dsc_tensor *DSC_RESTRICT x,
                                         const dsc_tensor *DSC_RESTRICT indexes);

extern void dsc_tensor_set_idx(dsc_ctx *ctx,
                               dsc_tensor *DSC_RESTRICT xa,
                               const dsc_tensor *DSC_RESTRICT xb,
                               int indexes...);

extern void dsc_tensor_set_slice(dsc_ctx *ctx,
                                 dsc_tensor *DSC_RESTRICT xa,
                                 const dsc_tensor *DSC_RESTRICT xb,
                                 int slices...);

// ============================================================
// Binary Operations

extern dsc_tensor *dsc_add(dsc_ctx *ctx,
                           dsc_tensor *xa,
                           dsc_tensor *xb,
                           dsc_tensor *out = nullptr);

extern dsc_tensor *dsc_sub(dsc_ctx *ctx,
                           dsc_tensor *xa,
                           dsc_tensor *xb,
                           dsc_tensor *out = nullptr);

extern dsc_tensor *dsc_mul(dsc_ctx *ctx,
                           dsc_tensor *xa,
                           dsc_tensor *xb,
                           dsc_tensor *out = nullptr);

extern dsc_tensor *dsc_div(dsc_ctx *ctx,
                           dsc_tensor *xa,
                           dsc_tensor *xb,
                           dsc_tensor *out = nullptr);

extern dsc_tensor *dsc_pow(dsc_ctx *ctx,
                           dsc_tensor *xa,
                           dsc_tensor *xb,
                           dsc_tensor *out = nullptr);

extern dsc_tensor *dsc_matmul(dsc_ctx *ctx,
                              dsc_tensor *DSC_RESTRICT xa,
                              dsc_tensor *DSC_RESTRICT xb,
                              bool trans_b = false,
                              dsc_tensor *DSC_RESTRICT out = nullptr);

extern dsc_tensor *dsc_compare(dsc_ctx *ctx,
                               const dsc_tensor *xa,
                               const dsc_tensor *xb,
                               dsc_comparison_op comp,
                               dsc_tensor *out = nullptr);

extern void dsc_masked_fill(dsc_ctx *ctx,
                            dsc_tensor *x,
                            const dsc_tensor *mask,
                            f64 value);

// ============================================================
// Unary Operations

extern dsc_tensor *dsc_cos(dsc_ctx *ctx,
                           dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out = nullptr);

extern dsc_tensor *dsc_sin(dsc_ctx *ctx,
                           dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out = nullptr);

extern dsc_tensor *dsc_tanh(dsc_ctx *ctx,
                            dsc_tensor *DSC_RESTRICT x,
                            dsc_tensor *DSC_RESTRICT out = nullptr);

extern dsc_tensor *dsc_exp(dsc_ctx *ctx,
                           dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out = nullptr);

extern dsc_tensor *dsc_sqrt(dsc_ctx *ctx,
                            dsc_tensor *DSC_RESTRICT x,
                            dsc_tensor *DSC_RESTRICT out = nullptr);

// ============================================================
// Unary Operations Along Axis

extern dsc_tensor *dsc_sum(dsc_ctx *ctx,
                           dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out = nullptr,
                           int axis = -1,
                           bool keep_dims = true);

extern dsc_tensor *dsc_max(dsc_ctx *ctx,
                           dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out = nullptr,
                           int axis = -1,
                           bool keep_dims = true);

extern dsc_tensor *dsc_min(dsc_ctx *ctx,
                           dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out = nullptr,
                           int axis = -1,
                           bool keep_dims = true);

#if defined(__cplusplus)
}
#endif
