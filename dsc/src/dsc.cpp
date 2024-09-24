// Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#include "dsc.h"
#include "dsc_ops.h"
#include "dsc_fft.h"
#include "dsc_iter.h"
#include <cstring>
#include <random>
#include <cstdarg> // va_xxx


// How many independent FFT plans we support. This is completely arbitrary
#if !defined(DSC_FFT_PLANS)
#   define DSC_FFT_PLANS ((int) 16)
#endif

#if !defined(DSC_PAGE_SIZE)
#   define DSC_PAGE_SIZE ((usize) 4096)
#endif

#define DSC_SIMD_ALIGN ((int) 32)

#define CONST_OP_IMPL(func, type, op) \
    dsc_tensor *dsc_##func##_##type(dsc_ctx *ctx,               \
                                    dsc_tensor *x,              \
                                    const type val,             \
                                    dsc_tensor *out) noexcept { \
        validate_unary_params();    \
        scalar_op(x, out, val, op); \
        return out;                 \
}

// This needs to be a macro otherwise the pointer assignments to out, xa and xb
// would not work unless I pass them as pointers to pointers which is very ugly.
#define validate_binary_params() \
    do {                                    \
        DSC_ASSERT(xa != nullptr);          \
        DSC_ASSERT(xb != nullptr);          \
        DSC_ASSERT(can_broadcast(xa, xb));  \
\
        const int n_dim = DSC_MAX(xa->n_dim, xb->n_dim); \
\
        int shape[DSC_MAX_DIMS];                                                                \
        for (int i = 0; i < DSC_MAX_DIMS; ++i) shape[i] = DSC_MAX(xa->shape[i], xb->shape[i]);  \
\
       const dsc_dtype out_dtype = DSC_DTYPE_CONVERSION_TABLE[xa->dtype][xb->dtype]; \
\
        if (out == nullptr) {                                                               \
            out = dsc_new_tensor(ctx, n_dim, &shape[DSC_MAX_DIMS - n_dim], out_dtype);      \
        } else {                                                                            \
            DSC_ASSERT(out->dtype == out_dtype);                                            \
            DSC_ASSERT(out->n_dim == n_dim);                                                \
            DSC_ASSERT(memcmp(out->shape, shape, DSC_MAX_DIMS * sizeof(shape[0])) == 0);    \
        }                                                                                   \
\
        xa = dsc_cast(ctx, xa, out_dtype);  \
        xb = dsc_cast(ctx, xb, out_dtype);  \
    } while (0)                             \

#define validate_unary_params() \
    do {                                \
        DSC_ASSERT(x != nullptr);       \
        if (out == nullptr) {           \
            out = dsc_new_like(ctx, x); \
        } else {                        \
            DSC_ASSERT(out->dtype == x->dtype);                                                     \
            DSC_ASSERT(out->n_dim == x->n_dim);                                                     \
            DSC_ASSERT(memcmp(out->shape, x->shape, DSC_MAX_DIMS * sizeof(out->shape[0])) == 0);    \
        }                                                                                           \
    } while(0)

#define validate_reduce_params()    \
    do {                            \
        DSC_ASSERT(x != nullptr);   \
\
        const int axis_idx = dsc_tensor_dim(x, axis);   \
        DSC_ASSERT(axis_idx < DSC_MAX_DIMS);            \
\
        int out_shape[DSC_MAX_DIMS];                    \
        int out_ndim = x->n_dim;                        \
        if (keep_dims) {                                \
            memcpy(out_shape, x->shape, DSC_MAX_DIMS * sizeof(*out_shape));                         \
            out_shape[axis_idx] = 1;                                                                \
        } else {                                                                                    \
            out_ndim--;                                                                             \
            const int out_offset = DSC_MAX_DIMS - out_ndim;                                         \
            memset(out_shape, 1, out_offset * sizeof(*out_shape));                                  \
            for (int x_idx = DSC_MAX_DIMS - x->n_dim, out_idx = 0; x_idx < DSC_MAX_DIMS; ++x_idx) { \
                if (x_idx == axis_idx)                              \
                    continue;                                       \
\
                out_shape[out_offset + out_idx] = x->shape[x_idx];  \
                out_idx++;                                          \
            }                                                       \
        }                                                           \
\
        if (out == nullptr) {                                                                   \
            out = dsc_new_tensor(ctx, out_ndim, &out_shape[DSC_MAX_DIMS - out_ndim], x->dtype); \
        } else {                                \
            DSC_ASSERT(out->dtype == x->dtype); \
            DSC_ASSERT(out->n_dim == out_ndim); \
            DSC_ASSERT(memcmp(out->shape, out_shape, DSC_MAX_DIMS * sizeof(*out_shape)) == 0);  \
        }                                                                                       \
    } while(0)

#define dsc_for(idx, X) for (int idx = 0; idx < (X)->ne; ++idx)

#define DSC_CTX_PUSH(CTX) \
    dsc_obj *checkpointed_last_ = (CTX)->buffer->last;      \
    const int checkpointed_n_objs_ = (CTX)->buffer->n_objs

#define DSC_CTX_POP(CTX) \
    (CTX)->buffer->last = checkpointed_last_;     \
    (CTX)->buffer->n_objs = checkpointed_n_objs_

struct dsc_obj {
    usize offset;
    usize nb;
};

struct dsc_buffer {
    dsc_fft_plan *plans[DSC_FFT_PLANS];
    dsc_obj *last;
    usize nb;
    int n_objs;
    int n_plans;
};

struct dsc_ctx {
    dsc_buffer *buffer;
};

// ============================================================
// Initialization

static DSC_MALLOC dsc_buffer *dsc_buffer_alloc(const usize nb) noexcept {
    const usize buff_size = DSC_ALIGN(nb + sizeof(dsc_buffer), DSC_PAGE_SIZE);

    dsc_buffer *buff = (dsc_buffer *) aligned_alloc(DSC_PAGE_SIZE, buff_size);
    DSC_ASSERT(buff != nullptr);

    buff->nb = buff_size - sizeof(dsc_buffer);
    buff->n_objs = 0;
    buff->n_plans = 0;
    buff->last = nullptr;

    return buff;
}

dsc_ctx *dsc_ctx_init(const usize nb) noexcept {
    DSC_ASSERT(nb > 0);

    dsc_ctx *ctx = (dsc_ctx *) malloc(sizeof(dsc_ctx));
    DSC_ASSERT(ctx != nullptr);

    ctx->buffer = dsc_buffer_alloc(nb);

    DSC_LOG_INFO("created new context %p of %ldMB",
                 (void *) ctx,
                 (usize) DSC_B_TO_MB(ctx->buffer->nb)
    );

    return ctx;
}

static dsc_fft_plan *dsc_get_plan(dsc_ctx *ctx, const int n,
                                  const dsc_fft_type fft_type,
                                  const dsc_dtype dtype) noexcept {
    dsc_dtype twd_dtype;
    switch (dtype) {
        case C32:
        case F32:
            twd_dtype = F32;
            break;
        case C64:
        case F64:
            twd_dtype = F64;
            break;
        DSC_INVALID_CASE("unknown dtype=%d", dtype);
    }

    const dsc_buffer *buffer = ctx->buffer;
    dsc_fft_plan *plan = nullptr;
    for (int i = 0; i < buffer->n_plans; ++i) {
        dsc_fft_plan *cached_plan = buffer->plans[i];
        if ((cached_plan != nullptr) &&
            (cached_plan->n == n) &&
            // Note: technically we could support complex FFTs of order N from real FFTs plans
            // but not the other way around because we need an extra set of twiddles in the RFFT.
            (cached_plan->fft_type == fft_type) &&
            (cached_plan->dtype == twd_dtype)) {
            plan = cached_plan;
            break;
        }
    }

    return plan;
}

template<typename T>
static DSC_MALLOC DSC_INLINE T *dsc_obj_alloc(dsc_buffer *buff, const usize nb) noexcept {
    const usize last_offset = buff->last == nullptr ? 0 : buff->last->offset;
    const usize last_size = buff->last == nullptr ? 0 : buff->last->nb;
    const usize last_end = last_offset + last_size;

    if (nb + sizeof(dsc_obj) + last_end > buff->nb) {
        DSC_LOG_FATAL("can't allocate %.2fKB", DSC_B_TO_KB(nb));
    }

    // The actual buffer starts after the 'header' of the arena struct.
    dsc_obj *new_obj = (dsc_obj *) ((byte *) buff + last_end + sizeof(dsc_buffer));
    // The offset refers to the actual offset of the "contained" object which comes after
    // the dsc_object "header".
    new_obj->offset = last_end + sizeof(dsc_obj);
    new_obj->nb = nb;

    buff->n_objs++;
    buff->last = new_obj;

    return (T *) ((byte *) buff + sizeof(dsc_buffer) + buff->last->offset);
}

dsc_fft_plan *dsc_plan_fft(dsc_ctx *ctx, const int n,
                           const dsc_fft_type fft_type,
                           const dsc_dtype dtype) noexcept {
    const int fft_n = dsc_fft_best_n(n);

    dsc_fft_plan *plan = dsc_get_plan(ctx, fft_n, fft_type, dtype);

    if (plan == nullptr) {
        dsc_buffer *buffer = ctx->buffer;
        if (buffer->n_plans <= DSC_FFT_PLANS) {
            const usize storage = sizeof(dsc_fft_plan) + dsc_fft_storage(fft_n, dtype, fft_type);

            DSC_LOG_DEBUG("allocating new %s plan with N=%d dtype=%s",
                          fft_type == REAL ? "RFFT" : "FFT",
                          fft_n, DSC_DTYPE_NAMES[dtype]);

            plan = dsc_obj_alloc<dsc_fft_plan>(buffer, storage);
            plan->twiddles = (plan + 1);
            dsc_init_plan(plan, fft_n, dtype, fft_type);

            buffer->plans[buffer->n_plans++] = plan;
        } else {
            DSC_LOG_FATAL("too many plans in context");
        }
    } else {
        DSC_LOG_DEBUG("found cached %s plan with N=%d dtype=%s",
                      fft_type == REAL ? "RFFT" : "FFT",
                      fft_n, DSC_DTYPE_NAMES[dtype]);
    }

    return plan;
}

// ============================================================
// Cleanup/Teardown

void dsc_ctx_free(dsc_ctx *ctx) noexcept {
    DSC_LOG_INFO("freeing context %p of %ldMB",
                 (void *) ctx,
                 (usize) DSC_B_TO_MB(ctx->buffer->nb)
    );

    free(ctx->buffer);
    free(ctx);
}

void dsc_ctx_clear(dsc_ctx *ctx) noexcept {
    DSC_LOG_DEBUG("clearing context %p: mem_size=%ldMB n_objs=%d fft_plans=%d",
                  (void *)ctx,
                  (usize) DSC_B_TO_MB(ctx->buffer->nb),
                  ctx->buffer->n_objs,
                  ctx->buffer->n_plans
    );
    ctx->buffer->last = nullptr;
    ctx->buffer->n_objs = 0;
    // Set all the current plans to nullptr
    for (int i = 0; i < ctx->buffer->n_plans; ++i)
        ctx->buffer->plans[i] = nullptr;

    ctx->buffer->n_plans = 0;
}

// ============================================================
// Tensor Creation

DSC_MALLOC dsc_tensor *dsc_new_tensor(dsc_ctx *ctx,
                                      const int n_dim,
                                      const int *shape,
                                      const dsc_dtype dtype) noexcept {
    DSC_ASSERT((unsigned) n_dim <= DSC_MAX_DIMS);

    int ne = 1;
    for (int i = 0; i < n_dim; ++i) ne *= shape[i];

    const usize mem_needed = sizeof(dsc_tensor) + ne * DSC_DTYPE_SIZE[dtype] + DSC_SIMD_ALIGN;

    dsc_buffer *buff = ctx->buffer;

    // Allocate the actual tensor
    dsc_tensor *new_tensor = dsc_obj_alloc<dsc_tensor>(buff, mem_needed);

    new_tensor->dtype = dtype;

    new_tensor->ne = ne;
    new_tensor->n_dim = n_dim;

    // If n_dim is lower than DSC_MAX_DIM then we need to pre-fill the beginning of the array with 1
    for (int i = 0; i < DSC_MAX_DIMS; ++i) {
        new_tensor->shape[i] = i < (DSC_MAX_DIMS - n_dim) ? 1 : shape[i - (DSC_MAX_DIMS - n_dim)];
    }

    // Compute the stride
    memset(new_tensor->stride, 0, DSC_MAX_DIMS * sizeof(int));
    // Todo: stride as number of bytes?
    new_tensor->stride[DSC_MAX_DIMS - 1] = 1;
    for (int i = DSC_MAX_DIMS - 2; i >= 0; --i) {
        new_tensor->stride[i] = new_tensor->stride[i + 1] * new_tensor->shape[i + 1];
    }

    const uintptr_t unaligned_offset = (uintptr_t) ((byte *) new_tensor + sizeof(dsc_tensor));
    new_tensor->data = (void *) (DSC_ALIGN(unaligned_offset, DSC_SIMD_ALIGN));

    DSC_LOG_DEBUG("n_dim=%d shape=[%d, %d, %d, %d] stride=[%d, %d, %d, %d] dtype=%s",
                  n_dim, new_tensor->shape[0], new_tensor->shape[1], new_tensor->shape[2],
                  new_tensor->shape[3], new_tensor->stride[0], new_tensor->stride[1], new_tensor->stride[2],
                  new_tensor->stride[3], DSC_DTYPE_NAMES[dtype]
    );

    return new_tensor;
}

dsc_tensor *dsc_tensor_1d(dsc_ctx *ctx, const dsc_dtype dtype,
                          const int dim1) noexcept {
    const int shape[DSC_MAX_DIMS] = {dim1};
    return dsc_new_tensor(ctx, 1, shape, dtype);
}

dsc_tensor *dsc_tensor_2d(dsc_ctx *ctx, const dsc_dtype dtype,
                          const int dim1, const int dim2) noexcept {
    const int shape[DSC_MAX_DIMS] = {dim1, dim2};
    return dsc_new_tensor(ctx, 2, shape, dtype);
}

dsc_tensor *dsc_tensor_3d(dsc_ctx *ctx, const dsc_dtype dtype,
                          const int dim1, const int dim2,
                          const int dim3) noexcept {
    const int shape[DSC_MAX_DIMS] = {dim1, dim2, dim3};
    return dsc_new_tensor(ctx, 3, shape, dtype);
}

dsc_tensor *dsc_tensor_4d(dsc_ctx *ctx, const dsc_dtype dtype,
                          const int dim1, const int dim2,
                          const int dim3, const int dim4) noexcept {
    const int shape[DSC_MAX_DIMS] = {dim1, dim2, dim3, dim4};
    return dsc_new_tensor(ctx, 4, shape, dtype);
}

template<typename T>
static DSC_INLINE void assign_op(dsc_tensor *DSC_RESTRICT x,
                                 const T start, const T step) noexcept {
    DSC_TENSOR_DATA(T, x);

    T val = start;
    dsc_for(i, x) {
        x_data[i] = val;
        val = add_op()(val, step);
    }
}

dsc_tensor *dsc_wrap_f32(dsc_ctx *ctx, const f32 val) noexcept {
    dsc_tensor *out = dsc_tensor_1d(ctx, F32, 1);

    DSC_TENSOR_DATA(f32, out);
    out_data[0] = val;

    return out;
}

dsc_tensor *dsc_wrap_f64(dsc_ctx *ctx, const f64 val) noexcept {
    dsc_tensor *out = dsc_tensor_1d(ctx, F64, 1);

    DSC_TENSOR_DATA(f64, out);
    out_data[0] = val;

    return out;
}

dsc_tensor *dsc_wrap_c32(dsc_ctx *ctx, const c32 val) noexcept {
    dsc_tensor *out = dsc_tensor_1d(ctx, C32, 1);

    DSC_TENSOR_DATA(c32, out);
    out_data[0] = val;

    return out;
}

dsc_tensor *dsc_wrap_c64(dsc_ctx *ctx, const c64 val) noexcept {
    dsc_tensor *out = dsc_tensor_1d(ctx, C64, 1);

    DSC_TENSOR_DATA(c64, out);
    out_data[0] = val;

    return out;
}

dsc_tensor *dsc_arange(dsc_ctx *ctx,
                       const int n,
                       const dsc_dtype dtype) noexcept {
    dsc_tensor *out = dsc_tensor_1d(ctx, dtype, n);
    switch (dtype) {
        case dsc_dtype::F32:
            assign_op<f32>(out, {}, 1);
            break;
        case dsc_dtype::F64:
            assign_op<f64>(out, {}, 1);
            break;
        case dsc_dtype::C32:
            assign_op<c32>(out, dsc_complex(c32, 0, 0), dsc_complex(c32, 1, 0));
            break;
        case dsc_dtype::C64:
            assign_op<c64>(out, dsc_complex(c64, 0, 0), dsc_complex(c64, 1, 0));
            break;
        DSC_INVALID_CASE("unknown dtype %d", dtype);
    }
    return out;
}

template<typename T>
static DSC_INLINE void dsc_fill_randn(dsc_tensor *x) noexcept {
    static_assert(dsc_is_real<T>(), "T must be real");

    DSC_TENSOR_DATA(T, x);

    std::mt19937 rng;
    std::normal_distribution<T> dist;

    for (int i = 0; i < x->ne; ++i)
        x_data[i] = dist(rng);
}

dsc_tensor *dsc_randn(dsc_ctx *ctx,
                      const int n_dim,
                      const int *shape,
                      const dsc_dtype dtype) noexcept {
    dsc_tensor *out = dsc_new_tensor(ctx, n_dim, shape, dtype);

    switch (out->dtype) {
        case F32:
            dsc_fill_randn<f32>(out);
            break;
        case F64:
            dsc_fill_randn<f64>(out);
            break;
        DSC_INVALID_CASE("dtype must be real");
    }

    return out;
}

template<typename Tx, typename To>
static DSC_INLINE void copy_op(const dsc_tensor *DSC_RESTRICT x,
                               dsc_tensor *DSC_RESTRICT out) noexcept {
    DSC_TENSOR_DATA(Tx, x);
    DSC_TENSOR_DATA(To, out);

    // Todo: I can probably do better but (if it works) it's fine for now
    dsc_for(i, out) {
        out_data[i] = cast_op().template operator()<Tx, To>(x_data[i]);
    }
}

template<typename Tx>
static DSC_INLINE void copy_op(const dsc_tensor *DSC_RESTRICT x,
                               dsc_tensor *DSC_RESTRICT out) noexcept {
    switch (out->dtype) {
        case dsc_dtype::F32:
            copy_op<Tx, f32>(x, out);
            break;
        case dsc_dtype::F64:
            copy_op<Tx, f64>(x, out);
            break;
        case dsc_dtype::C32:
            copy_op<Tx, c32>(x, out);
            break;
        case dsc_dtype::C64:
            copy_op<Tx, c64>(x, out);
            break;
        DSC_INVALID_CASE("unknown dtype %d", x->dtype);
    }
}

static DSC_INLINE void copy(const dsc_tensor *DSC_RESTRICT x,
                            dsc_tensor *DSC_RESTRICT out) noexcept {
    switch (x->dtype) {
        case dsc_dtype::F32:
            copy_op<f32>(x, out);
            break;
        case dsc_dtype::F64:
            copy_op<f64>(x, out);
            break;
        case dsc_dtype::C32:
            copy_op<c32>(x, out);
            break;
        case dsc_dtype::C64:
            copy_op<c64>(x, out);
            break;
        DSC_INVALID_CASE("unknown dtype %d", x->dtype);
    }
}

dsc_tensor *dsc_cast(dsc_ctx *ctx, dsc_tensor *DSC_RESTRICT x,
                     const dsc_dtype new_dtype) noexcept {
    if (x->dtype == new_dtype)
        return x;

    dsc_tensor *out = dsc_new_tensor(ctx, x->n_dim, &x->shape[DSC_MAX_DIMS - x->n_dim], new_dtype);
    copy(x, out);

    return out;
}

// ============================================================
// Indexing and Slicing
//

dsc_tensor *dsc_tensor_get_idx(dsc_ctx *ctx,
                               const dsc_tensor *DSC_RESTRICT x,
                               const int indexes...) noexcept {
    DSC_ASSERT(x != nullptr);
    DSC_ASSERT((unsigned) indexes <= DSC_MAX_DIMS);

    if (indexes > x->n_dim) {
        DSC_LOG_FATAL("too many indexes");
    }

    int el_idx[DSC_MAX_DIMS];

    std::va_list args;
    va_start(args, indexes);
    for (int i = 0; i < indexes; ++i) {
        int idx = va_arg(args, int);
        const int x_dim_i = x->shape[dsc_tensor_dim(x, i)];
        // Negative indexes mean accessing from the end
        if (idx < 0) idx += x_dim_i;

        DSC_ASSERT((unsigned) idx < (unsigned) x_dim_i);

        el_idx[i] = idx;
    }
    va_end(args);

    // Since we are wrapping scalars the resulting tensor will be always at least 1D
    const int out_n_dim = x->n_dim == indexes ? 1 : x->n_dim - indexes;
    // If we are indexing a single element then of course the output shape will be just 1
    int out_shape[DSC_MAX_DIMS] = {1};
    if (x->n_dim > indexes) {
        memcpy(out_shape, &x->shape[DSC_MAX_DIMS - out_n_dim], out_n_dim * sizeof(*x->shape));
    }

    dsc_tensor *out = dsc_new_tensor(ctx, out_n_dim, out_shape, x->dtype);

    int offset = 0;
    for (int i = 0; i < indexes; ++i) {
        offset += (x->stride[dsc_tensor_dim(x, i)] * el_idx[i]);
    }
    const int stride = x->stride[dsc_tensor_dim(x, (indexes - 1))];

    memcpy(out->data, ((byte *) x->data) + (offset * DSC_DTYPE_SIZE[x->dtype]), stride * DSC_DTYPE_SIZE[x->dtype]);

    return out;
}

template <typename T>
static DSC_INLINE void copy_slice(const dsc_tensor *DSC_RESTRICT x,
                                  dsc_tensor *DSC_RESTRICT out,
                                  const int n_slices,
                                  dsc_slice *slices) noexcept {
    DSC_TENSOR_DATA(T, out);
    DSC_TENSOR_DATA(T, x);
    dsc_slice_iterator x_it(x, n_slices, slices);

    dsc_for(i, out) {
        out_data[i] = x_data[x_it.index()];

        x_it.next();
    }
}

static DSC_INLINE void parse_slices(const dsc_tensor *DSC_RESTRICT x,
                                    dsc_slice *parsed_slices,
                                    bool *collapse_dim,
                                    const int slices,
                                    std::va_list args) noexcept {
    for (int i = 0; i < slices; ++i) {
        dsc_slice slice = va_arg(args, dsc_slice);
        const int x_dim_i = x->shape[dsc_tensor_dim(x, i)];

        // The convention is to set all fields in the slice to the same value != NONE to signal
        // access to a single index rather than a slice (happens in mixed scenarios like x[:, 1])
        if (slice.start == slice.stop &&
            slice.start == slice.step &&
            slice.start != DSC_SLICE_NONE) {
            // If we need to return a tensor then we need to keep track of the dimensions that must
            // be collapsed to match NumPy behaviour
            if (collapse_dim != nullptr) collapse_dim[i] = true;
            slice.step = 1;
            if (slice.start < 0) {
                slice.start += x_dim_i;
                slice.stop += x_dim_i + 1;
            } else {
                slice.stop += 1;
            }
        }

        DSC_ASSERT(slice.step != 0);

        // If a field is marked using DSC_SLICE_NONE then replace it with the 'default' behaviour.
        // The default behaviour is controlled by step (see: https://numpy.org/doc/stable/user/basics.indexing.html)
        if (slice.step == DSC_SLICE_NONE) slice.step = 1;
        if (slice.start == DSC_SLICE_NONE) {
            if (slice.step > 0) slice.start = 0;
            else slice.start = x_dim_i - 1;
        }
        if (slice.stop == DSC_SLICE_NONE) {
            if (slice.step > 0) slice.stop = x_dim_i;
            else slice.stop = -x_dim_i - 1;
        }

        if (slice.start < 0) slice.start += x_dim_i;
        if (slice.stop < 0) slice.stop += x_dim_i;

        DSC_ASSERT(abs(slice.stop - slice.start) <= x_dim_i);
        DSC_ASSERT((slice.step > 0 && slice.start < slice.stop) || (slice.step < 0 && slice.start > slice.stop));

        DSC_ASSERT(abs(slice.step) <= x_dim_i);

        parsed_slices[i] = slice;
    }
}

dsc_tensor *dsc_tensor_get_slice(dsc_ctx *ctx,
                                 const dsc_tensor *DSC_RESTRICT x,
                                 const int slices...) noexcept {
    DSC_ASSERT(x != nullptr);
    DSC_ASSERT((unsigned) slices <= DSC_MAX_DIMS);

    if (slices > x->n_dim) {
        DSC_LOG_FATAL("too many slices");
    }

    dsc_slice el_slices[DSC_MAX_DIMS];
    bool collapse_dim[DSC_MAX_DIMS] = {false};

    std::va_list args;
    va_start(args, slices);
    parse_slices(x, el_slices, collapse_dim, slices, args);
    va_end(args);

    int out_shape[DSC_MAX_DIMS];
    int out_n_dim = x->n_dim;
    for (int i = 0, out_idx = 0; i < x->n_dim; ++i) {
        if (i < slices) {
            if (collapse_dim[i]) {
                out_n_dim -= 1;
                continue;
            }
            const dsc_slice slice_i = el_slices[i];
            const int ne_i = abs(slice_i.stop - slice_i.start);
            const int abs_step = abs(slice_i.step);
            out_shape[out_idx] = (ne_i + abs_step - 1) / abs_step;
        } else {
            out_shape[out_idx] = x->shape[dsc_tensor_dim(x, i)];
        }
        out_idx += 1;
    }

    dsc_tensor *out = dsc_new_tensor(ctx, out_n_dim, out_shape, x->dtype);

    switch (out->dtype) {
        case F32:
            copy_slice<f32>(x, out, slices, el_slices);
            break;
        case F64:
            copy_slice<f64>(x, out, slices, el_slices);
            break;
        case C32:
            copy_slice<c32>(x, out, slices, el_slices);
            break;
        case C64:
            copy_slice<c64>(x, out, slices, el_slices);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }

    return out;
}

template <typename T>
static DSC_INLINE void tensor_set(dsc_tensor *DSC_RESTRICT xa,
                                  const bool xa_scalar,
                                  const dsc_tensor *DSC_RESTRICT xb,
                                  const int n_slices,
                                  const dsc_slice *slices) noexcept {
    DSC_TENSOR_DATA(T, xa);
    DSC_TENSOR_DATA(T, xb);

    if (xa_scalar) {
        int offset = 0;
        for (int i = 0; i < n_slices; ++i)
            offset += (slices[i].start * xa->stride[dsc_tensor_dim(xa, i)]);

        xa_data[offset] = xb_data[0];
    } else if (xb->n_dim == 1 && xb->shape[dsc_tensor_dim(xb, -1)] == 1) {
        const T el = xb_data[0];

        for (dsc_slice_iterator xa_it(xa, n_slices, slices);
             xa_it.has_next();
             xa_it.next()) {
            xa_data[xa_it.index()] = el;
        }
    } else {
        int xb_idx = 0;
        for (dsc_slice_iterator xa_it(xa, n_slices, slices);
             xa_it.has_next();
             xa_it.next()) {
            xa_data[xa_it.index()] = xb_data[xb_idx];

            xb_idx = (xb_idx + 1) % xb->ne;
        }
    }
}

void dsc_tensor_set_idx(dsc_ctx *,
                        dsc_tensor *DSC_RESTRICT xa,
                        const dsc_tensor *DSC_RESTRICT xb,
                        const int indexes...) noexcept {
    DSC_ASSERT(xa != nullptr);
    DSC_ASSERT(xb != nullptr);
    DSC_ASSERT((unsigned) indexes <= (unsigned) xa->n_dim);
    DSC_ASSERT(xa->dtype == xb->dtype);

    // Use slices so it's easier to iterate
    dsc_slice el_slices[DSC_MAX_DIMS];

    std::va_list args;
    va_start(args, indexes);
    for (int i = 0; i < indexes; ++i) {
        const int idx = va_arg(args, int);
        const int x_dim_i = xa->shape[dsc_tensor_dim(xa, i)];

        el_slices[i].start = idx;
        el_slices[i].stop = idx + 1;
        el_slices[i].step = 1;
        if (idx < 0) {
            el_slices[i].start += x_dim_i;
            el_slices[i].stop += x_dim_i;
        }
    }
    va_end(args);

    // If we do something like xa[2] and xa has more than one dimension then, the remaining
    // dimensions of xa and xb must be broadcastable together
    int xa_sub_shape[DSC_MAX_DIMS];
    for (int i = indexes; i < xa->n_dim; ++i)
        xa_sub_shape[i - indexes] = xa->shape[dsc_tensor_dim(xa, i - indexes)];

    const bool xb_scalar = xb->n_dim == 1 && xb->shape[dsc_tensor_dim(xb, -1)] == 1;
    const int xa_sub_ndim = xa->n_dim - indexes;

    if (xa_sub_ndim == 0) DSC_ASSERT(xb_scalar);

    if (!xb_scalar) {
        // If xb is not a scalar then its shape must be compatible with xa_sub_shape
        DSC_ASSERT(xb->n_dim == xa_sub_ndim);
        for (int i = 0; i < xa_sub_ndim; ++i) DSC_ASSERT(xa_sub_shape[i] == xb->shape[dsc_tensor_dim(xb, i)]);
    }

    switch (xa->dtype) {
        case dsc_dtype::F32:
            tensor_set<f32>(xa, xa_sub_ndim == 0, xb, indexes, el_slices);
            break;
        case dsc_dtype::F64:
            tensor_set<f64>(xa, xa_sub_ndim == 0, xb, indexes, el_slices);
            break;
        case dsc_dtype::C32:
            tensor_set<c32>(xa, xa_sub_ndim == 0, xb, indexes, el_slices);
            break;
        case dsc_dtype::C64:
            tensor_set<c64>(xa, xa_sub_ndim == 0, xb, indexes, el_slices);
            break;
        DSC_INVALID_CASE("unknown dtype %d", xa->dtype);
    }
}

void dsc_tensor_set_slice(dsc_ctx *,
                          dsc_tensor *DSC_RESTRICT xa,
                          const dsc_tensor *DSC_RESTRICT xb,
                          const int slices...) noexcept {
    DSC_ASSERT(xa != nullptr);
    DSC_ASSERT(xb != nullptr);
    DSC_ASSERT((unsigned) slices <= (unsigned) xa->n_dim);
    DSC_ASSERT(xa->dtype == xb->dtype);

    dsc_slice el_slices[DSC_MAX_DIMS];

    std::va_list args;
    va_start(args, slices);
    parse_slices(xa, el_slices, nullptr, slices, args);
    va_end(args);

    int xa_slice_shape[DSC_MAX_DIMS];
    for (int i = 0; i < xa->n_dim; ++i) {
        if (i < slices) {
            const dsc_slice slice_i = el_slices[i];
            const int ne_i = abs(slice_i.stop - slice_i.start);
            const int abs_step = abs(slice_i.step);
            xa_slice_shape[i] = (ne_i + abs_step - 1) / abs_step;
        } else {
            xa_slice_shape[i] = xa->shape[dsc_tensor_dim(xa, i)];
        }
    }

    const bool xb_scalar = xb->n_dim == 1 && xb->shape[dsc_tensor_dim(xb, -1)] == 1;

    if (!xb_scalar) {
        // Check whether xb is broadcastable with xa
        const int dims_to_compare = DSC_MIN(xa->n_dim, xb->n_dim);
        for (int i = 0; i < dims_to_compare; ++i) {
            const int xb_dim_i = xb->shape[dsc_tensor_dim(xb, i)];
            const int xa_slice_i = xa_slice_shape[i];
            DSC_ASSERT(xa_slice_i == 1 || xb_dim_i == 1 || xa_slice_i == xb_dim_i);
        }
    }

    bool xa_scalar = true;
    for (int i = 0; i < xa->n_dim && xa_scalar; ++i)
        xa_scalar &= xa_slice_shape[i] == 1;

    switch (xa->dtype) {
        case dsc_dtype::F32:
            tensor_set<f32>(xa, xa_scalar, xb, slices, el_slices);
            break;
        case dsc_dtype::F64:
            tensor_set<f64>(xa, xa_scalar, xb, slices, el_slices);
            break;
        case dsc_dtype::C32:
            tensor_set<c32>(xa, xa_scalar, xb, slices, el_slices);
            break;
        case dsc_dtype::C64:
            tensor_set<c64>(xa, xa_scalar, xb, slices, el_slices);
            break;
        DSC_INVALID_CASE("unknown dtype %d", xa->dtype);
    }
}

// ============================================================
// Binary Operations (Vector)

static bool DSC_INLINE DSC_PURE can_broadcast(const dsc_tensor *DSC_RESTRICT xa,
                                              const dsc_tensor *DSC_RESTRICT xb) noexcept {
    bool can_broadcast = true;
    for (int i = 0; i < DSC_MAX_DIMS && can_broadcast; ++i) {
        can_broadcast = xa->shape[i] == xb->shape[i] ||
                        xa->shape[i] == 1 ||
                        xb->shape[i] == 1;
    }

    return can_broadcast;
}

template<typename T, typename Op>
static DSC_INLINE void binary_op(const dsc_tensor *DSC_RESTRICT xa,
                                 const dsc_tensor *DSC_RESTRICT xb,
                                 dsc_tensor *DSC_RESTRICT out,
                                 Op op) noexcept {
    DSC_TENSOR_DATA(T, xa);
    DSC_TENSOR_DATA(T, xb);
    DSC_TENSOR_DATA(T, out);

    dsc_broadcast_iterator xa_it(xa, out->shape), xb_it(xb, out->shape);
    dsc_for(i, out) {
        out_data[i] = op(
                xa_data[xa_it.index()],
                xb_data[xb_it.index()]
        );
        xa_it.next(), xb_it.next();
    }
}

template<typename Op>
static DSC_INLINE void binary_op(const dsc_tensor *DSC_RESTRICT xa,
                                 const dsc_tensor *DSC_RESTRICT xb,
                                 dsc_tensor *DSC_RESTRICT out,
                                 Op op) noexcept {
    switch (out->dtype) {
        case dsc_dtype::F32:
            binary_op<f32>(xa, xb, out, op);
            break;
        case dsc_dtype::F64:
            binary_op<f64>(xa, xb, out, op);
            break;
        case dsc_dtype::C32:
            binary_op<c32>(xa, xb, out, op);
            break;
        case dsc_dtype::C64:
            binary_op<c64>(xa, xb, out, op);
            break;
        DSC_INVALID_CASE("unknown dtype %d", out->dtype);
    }
}

dsc_tensor *dsc_add(dsc_ctx *ctx,
                    dsc_tensor *DSC_RESTRICT xa,
                    dsc_tensor *DSC_RESTRICT xb,
                    dsc_tensor *DSC_RESTRICT out) noexcept {
    validate_binary_params();

    binary_op(xa, xb, out, add_op());

    return out;
}

dsc_tensor *dsc_sub(dsc_ctx *ctx,
                    dsc_tensor *DSC_RESTRICT xa,
                    dsc_tensor *DSC_RESTRICT xb,
                    dsc_tensor *DSC_RESTRICT out) noexcept {
    validate_binary_params();

    binary_op(xa, xb, out, sub_op());

    return out;
}

dsc_tensor *dsc_mul(dsc_ctx *ctx,
                    dsc_tensor *DSC_RESTRICT xa,
                    dsc_tensor *DSC_RESTRICT xb,
                    dsc_tensor *DSC_RESTRICT out) noexcept {
    validate_binary_params();

    binary_op(xa, xb, out, mul_op());

    return out;
}

dsc_tensor *dsc_div(dsc_ctx *ctx,
                    dsc_tensor *DSC_RESTRICT xa,
                    dsc_tensor *DSC_RESTRICT xb,
                    dsc_tensor *DSC_RESTRICT out) noexcept {
    validate_binary_params();

    binary_op(xa, xb, out, div_op());

    return out;
}

dsc_tensor *dsc_pow(dsc_ctx *ctx,
                    dsc_tensor *DSC_RESTRICT xa,
                    dsc_tensor *DSC_RESTRICT xb,
                    dsc_tensor *DSC_RESTRICT out) noexcept {
    validate_binary_params();

    binary_op(xa, xb, out, pow_op());

    return out;
}

// ============================================================
// Binary Operations (Scalar)

template<typename T, typename Op>
static DSC_INLINE void scalar_op(dsc_tensor *x,
                                 dsc_tensor *out,
                                 const T val,
                                 Op op) noexcept {

    T *x_data = (T *) x->data;
    T *out_data = (T *) out->data;

    dsc_for(i, out) {
        out_data[i] = op(x_data[i], val);
    }
}

CONST_OP_IMPL(addc, f32, add_op())
CONST_OP_IMPL(addc, f64, add_op())
CONST_OP_IMPL(addc, c32, add_op())
CONST_OP_IMPL(addc, c64, add_op())

CONST_OP_IMPL(subc, f32, sub_op())
CONST_OP_IMPL(subc, f64, sub_op())
CONST_OP_IMPL(subc, c32, sub_op())
CONST_OP_IMPL(subc, c64, sub_op())

CONST_OP_IMPL(mulc, f32, mul_op())
CONST_OP_IMPL(mulc, f64, mul_op())
CONST_OP_IMPL(mulc, c32, mul_op())
CONST_OP_IMPL(mulc, c64, mul_op())

CONST_OP_IMPL(divc, f32, div_op())
CONST_OP_IMPL(divc, f64, div_op())
CONST_OP_IMPL(divc, c32, div_op())
CONST_OP_IMPL(divc, c64, div_op())

CONST_OP_IMPL(powc, f32, pow_op())
CONST_OP_IMPL(powc, f64, pow_op())
CONST_OP_IMPL(powc, c32, pow_op())
CONST_OP_IMPL(powc, c64, pow_op())

// ============================================================
// Unary Operations

template<typename T, typename Op>
static DSC_INLINE void unary_op(const dsc_tensor *DSC_RESTRICT x,
                                dsc_tensor *DSC_RESTRICT out,
                                Op op) noexcept {
    DSC_TENSOR_DATA(T, x);
    DSC_TENSOR_DATA(T, out);

    dsc_for(i, out) {
        out_data[i] = op(x_data[i]);
    }
}

template<typename Op>
static DSC_INLINE void unary_op(const dsc_tensor *DSC_RESTRICT x,
                                dsc_tensor *DSC_RESTRICT out,
                                Op op) noexcept {
    switch (x->dtype) {
        case dsc_dtype::F32:
            unary_op<f32>(x, out, op);
            break;
        case dsc_dtype::F64:
            unary_op<f64>(x, out, op);
            break;
        case dsc_dtype::C32:
            unary_op<c32>(x, out, op);
            break;
        case dsc_dtype::C64:
            unary_op<c64>(x, out, op);
            break;
        DSC_INVALID_CASE("unknown dtype %d", x->dtype);
    }
}

dsc_tensor *dsc_cos(dsc_ctx *ctx,
                    const dsc_tensor *DSC_RESTRICT x,
                    dsc_tensor *DSC_RESTRICT out) noexcept {
    validate_unary_params();

    unary_op(x, out, cos_op());

    return out;
}

dsc_tensor *dsc_sin(dsc_ctx *ctx,
                    const dsc_tensor *DSC_RESTRICT x,
                    dsc_tensor *DSC_RESTRICT out) noexcept {
    validate_unary_params();
    
    unary_op(x, out, sin_op());

    return out;
}

dsc_tensor *dsc_sinc(dsc_ctx *ctx,
                     const dsc_tensor *DSC_RESTRICT x,
                     dsc_tensor *DSC_RESTRICT out) noexcept {
    validate_unary_params();

    unary_op(x, out, sinc_op());

    return out;
}

dsc_tensor *dsc_logn(dsc_ctx *ctx,
                     const dsc_tensor *DSC_RESTRICT x,
                     dsc_tensor *DSC_RESTRICT out) noexcept {
    validate_unary_params();

    unary_op(x, out, logn_op());

    return out;
}

dsc_tensor *dsc_log2(dsc_ctx *ctx,
                     const dsc_tensor *DSC_RESTRICT x,
                     dsc_tensor *DSC_RESTRICT out) noexcept {
    validate_unary_params();

    unary_op(x, out, log2_op());

    return out;
}

dsc_tensor *dsc_log10(dsc_ctx *ctx,
                      const dsc_tensor *DSC_RESTRICT x,
                      dsc_tensor *DSC_RESTRICT out) noexcept {
    validate_unary_params();

    unary_op(x, out, log10_op());

    return out;
}

dsc_tensor *dsc_exp(dsc_ctx *ctx,
                    const dsc_tensor *DSC_RESTRICT x,
                    dsc_tensor *DSC_RESTRICT out) noexcept {
    validate_unary_params();

    unary_op(x, out, exp_op());

    return out;
}

dsc_tensor *dsc_sqrt(dsc_ctx *ctx,
                     const dsc_tensor *DSC_RESTRICT x,
                     dsc_tensor *DSC_RESTRICT out) noexcept {
    validate_unary_params();

    unary_op(x, out, sqrt_op());

    return out;
}

static DSC_INLINE DSC_STRICTLY_PURE dsc_dtype as_real(const dsc_dtype dtype) noexcept {
    switch (dtype) {
        case F64:
        case C64:
            return F64;
        case F32:
        case C32:
            return F32;
        DSC_INVALID_CASE("unknown dtype=%d", dtype);
    }
}

template <typename Tx, typename To, typename Op>
static DSC_INLINE void complex_binary(const dsc_tensor *DSC_RESTRICT x,
                                      dsc_tensor *DSC_RESTRICT out,
                                      Op op) noexcept {
    DSC_TENSOR_DATA(Tx, x);
    DSC_TENSOR_DATA(To, out);

    dsc_for(i, x) {
        out_data[i] = op(x_data[i]);
    }
}

dsc_tensor *dsc_abs(dsc_ctx *ctx,
                    const dsc_tensor *DSC_RESTRICT x,
                    dsc_tensor *DSC_RESTRICT out) noexcept {
    DSC_ASSERT(x != nullptr);

    const dsc_dtype out_dtype = as_real(x->dtype);
    if (out == nullptr) {
        out = dsc_new_tensor(ctx, x->n_dim, &x->shape[DSC_MAX_DIMS - x->n_dim], out_dtype);
    } else {
        DSC_ASSERT(out->dtype == out_dtype);
        DSC_ASSERT(out->n_dim == x->n_dim);
        DSC_ASSERT(memcmp(out->shape, x->shape, DSC_MAX_DIMS * sizeof(out->shape[0])) == 0);
    }

    switch (x->dtype) {
        case F32:
            complex_binary<f32, f32>(x, out, abs_op());
            break;
        case F64:
            complex_binary<f64, f64>(x, out, abs_op());
            break;
        case C32:
            complex_binary<c32, f32>(x, out, abs_op());
            break;
        case C64:
            complex_binary<c64, f64>(x, out, abs_op());
            break;
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }

    return out;
}

dsc_tensor *dsc_angle(dsc_ctx *ctx,
                      const dsc_tensor *DSC_RESTRICT x) noexcept {
    DSC_ASSERT(x != nullptr);

    const dsc_dtype out_dtype = as_real(x->dtype);
    dsc_tensor *out = dsc_new_tensor(ctx, x->n_dim, &x->shape[DSC_MAX_DIMS - x->n_dim], out_dtype);

    switch (x->dtype) {
        case F32:
            complex_binary<f32, f32>(x, out, atan2_op());
            break;
        case F64:
            complex_binary<f64, f64>(x, out, atan2_op());
            break;
        case C32:
            complex_binary<c32, f32>(x, out, atan2_op());
            break;
        case C64:
            complex_binary<c64, f64>(x, out, atan2_op());
            break;
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }

    return out;
}

template <typename T, typename Op>
static DSC_INLINE void complex_unary(const dsc_tensor *DSC_RESTRICT x,
                                     dsc_tensor *DSC_RESTRICT out,
                                     Op op) noexcept {
    DSC_TENSOR_DATA(T, x);
    DSC_TENSOR_DATA(T, out);

    dsc_for(i, x) {
        out_data[i] = op(x_data[i]);
    }
}

dsc_tensor *dsc_conj(dsc_ctx *ctx,
                     dsc_tensor *DSC_RESTRICT x) noexcept {
    DSC_ASSERT(x != nullptr);

    if (x->dtype == F32 || x->dtype == F64) {
        DSC_LOG_DEBUG("the input is real so it will be returned as is");
        return x;
    }

    dsc_tensor *out = dsc_new_like(ctx, x);

    switch (x->dtype) {
        case C32:
            complex_unary<c32>(x, out, conj_op());
            break;
        case C64:
            complex_unary<c64>(x, out, conj_op());
            break;
        DSC_INVALID_CASE("dtype must be complex");
    }

    return out;
}

dsc_tensor *dsc_real(dsc_ctx *ctx,
                     dsc_tensor *DSC_RESTRICT x) noexcept {
    DSC_ASSERT(x != nullptr);

    if (x->dtype == F32 || x->dtype == F64) {
        DSC_LOG_DEBUG("the input is real so it will be returned as is");
        return x;
    }

    const dsc_dtype out_dtype = as_real(x->dtype);
    dsc_tensor *out = dsc_new_tensor(ctx, x->n_dim, &x->shape[DSC_MAX_DIMS - x->n_dim], out_dtype);

    switch (x->dtype) {
        case C32:
            complex_binary<c32, f32>(x, out, real_op());
            break;
        case C64:
            complex_binary<c64, f64>(x, out, real_op());
            break;
        DSC_INVALID_CASE("dtype must be complex");
    }

    return out;
}

dsc_tensor *dsc_imag(dsc_ctx *ctx,
                     const dsc_tensor *DSC_RESTRICT x) noexcept {
    DSC_ASSERT(x != nullptr);

    const dsc_dtype out_dtype = as_real(x->dtype);
    dsc_tensor *out = dsc_new_tensor(ctx, x->n_dim, &x->shape[DSC_MAX_DIMS - x->n_dim], out_dtype);

    switch (x->dtype) {
        case F32:
            complex_binary<f32, f32>(x, out, imag_op());
            break;
        case F64:
            complex_binary<f64, f64>(x, out, imag_op());
            break;
        case C32:
            complex_binary<c32, f32>(x, out, imag_op());
            break;
        case C64:
            complex_binary<c64, f64>(x, out, imag_op());
            break;
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }

    return out;
}

template <typename T>
static DSC_INLINE T i0(const T x) noexcept {
    static_assert(dsc_is_real<T>(), "T must be real");

    // Taken from Numerical Recipes
    if constexpr (dsc_is_type<T, f32>()) {
        f32 ax, y, res;
        if ((ax = fabsf(x)) < 3.75f) {
            y = x / 3.75f;
            y *= y;
            res = 1.f + y * (3.5156229f + y *
                            (3.0899424f + y *
                            (1.2067492f + y *
                            (0.2659732f + y *
                            (0.360768e-1f + y *
                            (0.45813e-2f)))))
            );
        } else {
            y = 3.75f / ax;
            res = (expf(ax) / sqrtf(ax)) *
                  (0.39894228f + y *
                  (0.1328592e-1f + y *
                  (0.225319e-2f + y *
                  (-0.157565e-2f + y *
                  (0.916281e-2f + y *
                  (-0.2057706e-1f + y *
                  (0.2635537e-1f + y *
                  (-0.1647633e-1f + y *
                  (0.392377e-2f))))))))
            );
        }

        return res;
    } else {
        f64 ax, y, res;
        if ((ax = fabs(x)) < 3.75) {
            y = x / 3.75;
            y *= y;
            res = 1. + y * (3.5156229 + y *
                           (3.0899424 + y *
                           (1.2067492 + y *
                           (0.2659732 + y *
                           (0.360768e-1 + y *
                           (0.45813e-2)))))
            );
        } else {
            y = 3.75 / ax;
            res = (exp(ax) / sqrt(ax)) *
                  (0.39894228 + y *
                  (0.1328592e-1 + y *
                  (0.225319e-2 + y *
                  (-0.157565e-2 + y *
                  (0.916281e-2 + y *
                  (-0.2057706e-1 + y *
                  (0.2635537e-1 + y *
                  (-0.1647633e-1 + y *
                  (0.392377e-2))))))))
            );
        }

        return res;
    }
}

template <typename T>
static DSC_INLINE void dsc_internal_i0(const dsc_tensor *DSC_RESTRICT x,
                                       dsc_tensor *DSC_RESTRICT out) noexcept {
    static_assert(dsc_is_real<T>(), "T must be real");

    DSC_TENSOR_DATA(T, x);
    DSC_TENSOR_DATA(T, out);

    dsc_for(i, out) {
        out_data[i] = i0(x_data[i]);
    }
}

dsc_tensor *dsc_i0(dsc_ctx *ctx,
                   const dsc_tensor *DSC_RESTRICT x) noexcept {
    DSC_ASSERT(x != nullptr);
    DSC_ASSERT(x->dtype == F32 || x->dtype == F64);

    dsc_tensor *out = dsc_new_like(ctx, x);

    switch (x->dtype) {
        case F32:
            dsc_internal_i0<f32>(x, out);
            break;
        case F64:
            dsc_internal_i0<f64>(x, out);
            break;
        DSC_INVALID_CASE("dtype must be real");
    }

    return out;
}

template<typename T>
static DSC_INLINE void clip(const dsc_tensor *DSC_RESTRICT x,
                            dsc_tensor *DSC_RESTRICT out,
                            const T x_min, const T x_max) noexcept {
    DSC_TENSOR_DATA(T, x);
    DSC_TENSOR_DATA(T, out);

    dsc_for(i, out) {
        out_data[i] = min_op()(
                max_op()(x_data[i], x_min),
                x_max);
    }
}

dsc_tensor *dsc_clip(dsc_ctx *ctx, const dsc_tensor *DSC_RESTRICT x,
                     dsc_tensor *DSC_RESTRICT out,
                     const f64 x_min, const f64 x_max) noexcept {
    DSC_ASSERT(x != nullptr);

    validate_unary_params();

    switch (out->dtype) {
        case F32:
            clip<f32>(x, out, (f32) x_min, (f32) x_max);
            break;
        case F64:
            clip<f64>(x, out, x_min, x_max);
            break;
        case C32:
            clip<c32>(x, out,
                      dsc_complex(c32, (f32) x_min, dsc_zero<f32>()),
                      dsc_complex(c32, (f32) x_max, dsc_zero<f32>())
            );
            break;
        case C64:
            clip<c64>(x, out,
                      dsc_complex(c64, x_min, dsc_zero<f64>()),
                      dsc_complex(c64, x_max, dsc_zero<f64>())
            );
            break;
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }

    return out;
}

// ============================================================
// Unary Operations Along Axis

template <typename T>
static DSC_INLINE void sum(const dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out,
                           int axis_idx) noexcept {
    DSC_TENSOR_DATA(T, x);
    DSC_TENSOR_DATA(T, out);

    const int axis_n = x->shape[axis_idx];
    dsc_axis_iterator x_it(x, axis_idx, axis_n);
    dsc_for(i, out) {
        T acc = dsc_zero<T>();
        for (int j = 0; j < axis_n; ++j) {
            acc = add_op()(acc, x_data[x_it.index()]);
            x_it.next();
        }
        out_data[i] = acc;
    }
}

dsc_tensor *dsc_sum(dsc_ctx *ctx,
                    const dsc_tensor *DSC_RESTRICT x,
                    dsc_tensor *DSC_RESTRICT out,
                    const int axis,
                    const bool keep_dims) noexcept {
    // Fixme: keepdims=false won't work if x->n_dim = 1 because a scalar cannot be returned
    //  from this function, for now probably it makes sense to emulate this in Python

    validate_reduce_params();

    const int axis_idx = dsc_tensor_dim(x, axis);

    switch (out->dtype) {
        case F32:
            sum<f32>(x, out, axis_idx);
            break;
        case F64:
            sum<f64>(x, out, axis_idx);
            break;
        case C32:
            sum<c32>(x, out, axis_idx);
            break;
        case C64:
            sum<c64>(x, out, axis_idx);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }

    return out;
}

dsc_tensor *dsc_mean(dsc_ctx *ctx,
                     const dsc_tensor *DSC_RESTRICT x,
                     dsc_tensor *DSC_RESTRICT out,
                     const int axis,
                     const bool keep_dims) noexcept {
    out = dsc_sum(ctx, x, out, axis, keep_dims);

    const int axis_idx = dsc_tensor_dim(x, axis);
    const int axis_n = x->shape[axis_idx];

    switch (out->dtype) {
        case F32: {
            const f32 scale = 1.f / (f32) axis_n;
            out = dsc_mulc_f32(ctx, out, scale, out);
            break;
        }
        case F64: {
            const f64 scale = 1. / (f64) axis_n;
            out = dsc_mulc_f64(ctx, out, scale, out);
            break;
        }
        case C32: {
            const c32 scale = dsc_complex(c32, 1.f / (f32) axis_n, 0.f);
            out = dsc_mulc_c32(ctx, out, scale, out);
            break;
        }
        case C64: {
            const c64 scale = dsc_complex(c64, 1. / (f64) axis_n, 0.);
            out = dsc_mulc_c64(ctx, out, scale, out);
            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }

    return out;
}

template <typename T>
static DSC_INLINE void max(const dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out,
                           int axis_idx) noexcept {
    DSC_TENSOR_DATA(T, x);
    DSC_TENSOR_DATA(T, out);

    const int axis_n = x->shape[axis_idx];
    dsc_axis_iterator x_it(x, axis_idx, axis_n);
    dsc_for(i, out) {
        T max = dsc_inf<T, false>();
        for (int j = 0; j < axis_n; ++j) {
            max = max_op()(max, x_data[x_it.index()]);
            x_it.next();
        }
        out_data[i] = max;
    }
}

dsc_tensor *dsc_max(dsc_ctx *ctx,
                     const dsc_tensor *DSC_RESTRICT x,
                     dsc_tensor *DSC_RESTRICT out,
                     const int axis,
                     const bool keep_dims) noexcept {

    validate_reduce_params();

    const int axis_idx = dsc_tensor_dim(x, axis);

    switch (out->dtype) {
        case F32:
            max<f32>(x, out, axis_idx);
            break;
        case F64:
            max<f64>(x, out, axis_idx);
            break;
        case C32:
            max<c32>(x, out, axis_idx);
            break;
        case C64:
            max<c64>(x, out, axis_idx);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }

    return out;
}

template <typename T>
static DSC_INLINE void min(const dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out,
                           int axis_idx) noexcept {
    DSC_TENSOR_DATA(T, x);
    DSC_TENSOR_DATA(T, out);

    const int axis_n = x->shape[axis_idx];
    dsc_axis_iterator x_it(x, axis_idx, axis_n);
    dsc_for(i, out) {
        T min = dsc_inf<T, true>();
        for (int j = 0; j < axis_n; ++j) {
            min = min_op()(min, x_data[x_it.index()]);
            x_it.next();
        }
        out_data[i] = min;
    }
}

dsc_tensor *dsc_min(dsc_ctx *ctx,
                     const dsc_tensor *DSC_RESTRICT x,
                     dsc_tensor *DSC_RESTRICT out,
                     const int axis,
                     const bool keep_dims) noexcept {

    validate_reduce_params();

    const int axis_idx = dsc_tensor_dim(x, axis);

    switch (out->dtype) {
        case F32:
            min<f32>(x, out, axis_idx);
            break;
        case F64:
            min<f64>(x, out, axis_idx);
            break;
        case C32:
            min<c32>(x, out, axis_idx);
            break;
        case C64:
            min<c64>(x, out, axis_idx);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }

    return out;
}

// ============================================================
// Fourier Transforms

template<typename Tin, typename Tout, bool forward>
static DSC_INLINE void exec_fft(dsc_ctx *ctx,
                                const dsc_tensor *DSC_RESTRICT x,
                                dsc_tensor *DSC_RESTRICT out,
                                const int axis, const int x_n,
                                const int fft_n) noexcept {
    const dsc_dtype out_dtype = out->dtype;
    dsc_fft_plan *plan = dsc_plan_fft(ctx, fft_n, COMPLEX, out_dtype);

    DSC_CTX_PUSH(ctx);
    // Push the arena to make these two buffers temporary
    dsc_tensor *buff = dsc_tensor_1d(ctx, out_dtype, fft_n);
    dsc_tensor *fft_work = dsc_tensor_1d(ctx, out_dtype, fft_n);

    DSC_TENSOR_DATA(Tin, x);
    DSC_TENSOR_DATA(Tout, buff);
    DSC_TENSOR_DATA(Tout, out);
    DSC_TENSOR_DATA(Tout, fft_work);

    dsc_axis_iterator x_it(x, axis, fft_n);
    dsc_axis_iterator out_it(out, axis, fft_n);

    while (x_it.has_next()) {
        for (int i = 0; i < fft_n; ++i) {
            if (i < x_n) {
                int idx = x_it.index();
                if constexpr (dsc_is_type<Tin, Tout>()) {
                    buff_data[i] = x_data[idx];
                } else {
                    buff_data[i] = cast_op().template operator()<Tin, Tout>(x_data[idx]);
                }

                x_it.next();
            } else {
                buff_data[i] = dsc_zero<Tout>();
            }
        }

        dsc_complex_fft<Tout, forward>(plan, buff_data, fft_work_data);

        for (int i = 0; i < fft_n; ++i) {
            const int idx = out_it.index();
            out_data[idx] = buff_data[i];

            out_it.next();
        }
    }

    DSC_CTX_POP(ctx);
}

template<bool forward>
static DSC_INLINE dsc_tensor *dsc_internal_fft(dsc_ctx *ctx,
                                               const dsc_tensor *DSC_RESTRICT x,
                                               dsc_tensor *DSC_RESTRICT out,
                                               int n,
                                               const int axis) noexcept {
    DSC_ASSERT(x != nullptr);

    const int axis_idx = dsc_tensor_dim(x, axis);
    DSC_ASSERT(axis_idx < DSC_MAX_DIMS);

    const int x_n = x->shape[axis_idx];
    const int axis_n = dsc_fft_best_n(x_n);
    if (n > 0) {
        n = dsc_fft_best_n(n);
    } else {
        n = axis_n;
    }

    int out_shape[DSC_MAX_DIMS];
    for (int i = 0; i < DSC_MAX_DIMS; ++i)
        out_shape[i] = i != axis_idx ? x->shape[i] : n;

    dsc_dtype out_dtype = x->dtype;
    if (x->dtype == F32) {
        out_dtype = C32;
    } else if (x->dtype == F64) {
        out_dtype = C64;
    }

    if (out == nullptr) {
        out = dsc_new_tensor(ctx, x->n_dim, &out_shape[DSC_MAX_DIMS - x->n_dim], out_dtype);
    } else {
        DSC_ASSERT(out->dtype == out_dtype);
        DSC_ASSERT(out->n_dim == x->n_dim);
        DSC_ASSERT(memcmp(out_shape, out->shape, DSC_MAX_DIMS * sizeof(out->shape[0])) == 0);
    }

    DSC_LOG_DEBUG("performing %s FFT of length %d on x=[%d %d %d %d] over axis %d with size %d",
                  forward ? "FWD" : "BWD", n,
                  x->shape[0], x->shape[1], x->shape[2], x->shape[3],
                  axis_idx, x->shape[axis_idx]);

    switch (x->dtype) {
        case F32:
            exec_fft<f32, c32, forward>(ctx, x, out, axis_idx, x_n, n);
            break;
        case F64:
            exec_fft<f64, c64, forward>(ctx, x, out, axis_idx, x_n, n);
            break;
        case C32:
            exec_fft<c32, c32, forward>(ctx, x, out, axis_idx, x_n, n);
            break;
        case C64:
            exec_fft<c64, c64, forward>(ctx, x, out, axis_idx, x_n, n);
            break;
        DSC_INVALID_CASE("unknown dtype %d", x->dtype);
    }

    return out;
}

dsc_tensor *dsc_fft(dsc_ctx *ctx,
                    const dsc_tensor *DSC_RESTRICT x,
                    dsc_tensor *DSC_RESTRICT out,
                    const int n,
                    const int axis) noexcept {
    // Find N

    // Get the plan

    // --> Parallel START <--
    // Copy the desired axis of X over to a temp buffer (or out if we can go in-place?)

    // Perform the FFT

    // Write the result back (or not if we are working in-place?)
    // --> Parallel STOP <--

    // Done!
    return dsc_internal_fft<true>(ctx, x, out, n, axis);
}

dsc_tensor *dsc_ifft(dsc_ctx *ctx,
                     const dsc_tensor *DSC_RESTRICT x,
                     dsc_tensor *DSC_RESTRICT out,
                     const int n,
                     const int axis) noexcept {
    return dsc_internal_fft<false>(ctx, x, out, n, axis);
}

template<typename T, bool forward>
static DSC_INLINE void exec_rfft(dsc_ctx *ctx,
                                 const dsc_tensor *DSC_RESTRICT x,
                                 dsc_tensor *DSC_RESTRICT out,
                                 const int axis, const int x_n,
                                 const int out_n, const int fft_order) noexcept {
    static_assert(dsc_is_complex<T>(), "T must be the complex type of the transform");

    const dsc_dtype out_dtype = out->dtype;
    dsc_fft_plan *plan = dsc_plan_fft(ctx, fft_order, REAL, out_dtype);

    DSC_CTX_PUSH(ctx);
    // Push the arena to make these two buffers temporary
    dsc_tensor *buff = dsc_tensor_1d(ctx, out_dtype, out_n);
    dsc_tensor *fft_work = dsc_tensor_1d(ctx, out_dtype, out_n);

    DSC_TENSOR_DATA(T, fft_work);

    if constexpr (forward) {
        dsc_axis_iterator x_it(x, axis, DSC_MIN(x_n, fft_order * 2));
        dsc_axis_iterator out_it(out, axis, out_n);

        while (x_it.has_next()) {
            for (int i = 0; i < (fft_order << 1); ++i) {
                if (i < x_n) {
                    int idx = x_it.index();
                    ((real<T> *) buff->data)[i] = ((real<T> *) x->data)[idx];
                    x_it.next();
                } else {
                    ((real<T> *) buff->data)[i] = dsc_zero<real<T>>();
                }
            }

            dsc_real_fft<T, true>(plan, (T *)buff->data, fft_work_data);

            for (int i = 0; i < out_n; ++i) {
                const int idx = out_it.index();
                ((T *) out->data)[idx] = ((T *)buff->data)[i];

                out_it.next();
            }
        }
    } else {
        dsc_axis_iterator x_it(x, axis, fft_order + 1);
        dsc_axis_iterator out_it(out, axis, out_n);

        while (x_it.has_next()) {
            for (int i = 0; i < fft_order + 1; ++i) {
                if (i < x_n) {
                    int idx = x_it.index();
                    ((T *) buff->data)[i] = ((T *) x->data)[idx];
                    x_it.next();
                } else {
                    ((T *) buff->data)[i] = dsc_zero<T>();
                }
            }

            dsc_real_fft<T, false>(plan, (T *) buff->data, fft_work_data);

            for (int i = 0; i < out_n; ++i) {
                const int idx = out_it.index();
                ((real<T> *) out->data)[idx] = ((real<T> *) buff->data)[i];

                out_it.next();
            }
        }
    }


    DSC_CTX_POP(ctx);
}

template<bool forward>
static DSC_INLINE dsc_tensor *dsc_internal_rfft(dsc_ctx *ctx,
                                                const dsc_tensor *DSC_RESTRICT x,
                                                dsc_tensor *DSC_RESTRICT out,
                                                const int n,
                                                const int axis) noexcept {
    // For an RFFT if N is not specified N = (dim / 2) + 1
    // For an IRFFT if N is not specified N = 2 * (dim - 1)
    // Note: for now, since we support only power of 2 FFTs, the input of IRFFT is assumed to have
    //       the same shape as the output of RFFT. If this is not the case be careful, there can be
    //       issues with the results of IRFFT.
    DSC_ASSERT(x != nullptr);

    const int axis_idx = dsc_tensor_dim(x, axis);
    DSC_ASSERT(axis_idx < DSC_MAX_DIMS);

    const int x_n = x->shape[axis_idx];
    int out_n, fft_order;

    if constexpr (forward) {
        fft_order = ((n > 0) ? dsc_fft_best_n(n) : dsc_fft_best_n(x_n)) >> 1;
        out_n = fft_order + 1;
    } else {
        // Todo: verify that this makes sense
        fft_order = (n > 0) ? dsc_fft_best_n(n - 1) : dsc_fft_best_n(x_n - 1);
        out_n = fft_order << 1;
    }

    int out_shape[DSC_MAX_DIMS];
    for (int i = 0; i < DSC_MAX_DIMS; ++i)
        out_shape[i] = i != axis_idx ? x->shape[i] : out_n;

    dsc_dtype out_dtype;
    if constexpr (forward) {
        if (x->dtype == F32) out_dtype = C32;
        else if (x->dtype == F64) out_dtype = C64;
        else DSC_LOG_FATAL("RFFT input must be real");
    } else {
        if (x->dtype == C32) out_dtype = F32;
        else if (x->dtype == C64) out_dtype = F64;
        else DSC_LOG_FATAL("IRFFT input must be complex");
    }

    if (out == nullptr) {
        out = dsc_new_tensor(ctx, x->n_dim, &out_shape[DSC_MAX_DIMS - x->n_dim], out_dtype);
    } else {
        DSC_ASSERT(out->dtype == out_dtype);
        DSC_ASSERT(out->n_dim == x->n_dim);
        DSC_ASSERT(memcmp(out_shape, out->shape, DSC_MAX_DIMS * sizeof(out->shape[0])) == 0);
    }

    DSC_LOG_DEBUG("performing %s RFFT of length %d on x=[%d %d %d %d] over axis %d with size %d",
                  forward ? "FWD" : "BWD", n,
                  x->shape[0], x->shape[1], x->shape[2], x->shape[3],
                  axis_idx, x->shape[axis_idx]);

    switch (x->dtype) {
        case F32:
        case C32:
            exec_rfft<c32, forward>(ctx, x, out, axis_idx, x_n, out_n, fft_order);
            break;
        case F64:
        case C64:
            exec_rfft<c64, forward>(ctx, x, out, axis_idx, x_n, out_n, fft_order);
            break;
        DSC_INVALID_CASE("unknown dtype %d", x->dtype);
    }

    return out;
}

dsc_tensor *dsc_rfft(dsc_ctx *ctx,
                     const dsc_tensor *DSC_RESTRICT x,
                     dsc_tensor *DSC_RESTRICT out,
                     const int n,
                     const int axis) noexcept {
    return dsc_internal_rfft<true>(ctx, x, out, n, axis);
}

dsc_tensor *dsc_irfft(dsc_ctx *ctx,
                      const dsc_tensor *DSC_RESTRICT x,
                      dsc_tensor *DSC_RESTRICT out,
                      const int n,
                      const int axis) noexcept {
    return dsc_internal_rfft<false>(ctx, x, out, n, axis);
}

template<typename T>
static DSC_INLINE void dsc_internal_fftfreq(dsc_tensor *x,
                                           const int n,
                                           const T d) noexcept {
    static_assert(dsc_is_real<T>(), "T must be real");
    const T factor = 1 / (n * d);

    DSC_TENSOR_DATA(T, x);

    const int odd = n & 1;
    const int n2 = odd ? ((n - 1) >> 1) : (n >> 1);

    for (int i = 0; i < (n2 + odd); ++i)
        x_data[i] = i * factor;

    for (int i = 0; i < n2; ++i)
        x_data[(n2 + odd) + i] = (-n2 + i) * factor;

}

dsc_tensor *dsc_fftfreq(dsc_ctx *ctx,
                        const int n,
                        const f64 d,
                        const dsc_dtype dtype) noexcept {
    DSC_ASSERT(n > 0);

    // out = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
    // out = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

    dsc_tensor *out = dsc_tensor_1d(ctx, dtype, n);
    switch (dtype) {
        case F32:
            dsc_internal_fftfreq(out, n, (f32) d);
            break;
        case F64:
            dsc_internal_fftfreq(out, n, (f64) d);
            break;
        DSC_INVALID_CASE("dtype must be real");
    }

    return out;
}

template<typename T>
static DSC_INLINE void dsc_internal_rfftfreq(dsc_tensor *x,
                                            const int n,
                                            const T d) noexcept {
    static_assert(dsc_is_real<T>(), "T must be real");
    const T factor = 1 / (n * d);

    DSC_TENSOR_DATA(T, x);

    for (int i = 0; i < x->ne; ++i) {
        x_data[i] = i * factor;
    }
}

dsc_tensor *dsc_rfftfreq(dsc_ctx *ctx,
                         const int n,
                         const f64 d,
                         const dsc_dtype dtype) noexcept {
    DSC_ASSERT(n > 0);
    // out = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even
    // out = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n)   if n is odd
    // Note that the value of n that multiplies d is the same in both cases.
    const int n2 = (n & 1) ? (((n - 1) >> 1) + 1) : ((n >> 1) + 1);

    dsc_tensor *out = dsc_tensor_1d(ctx, dtype, n2);
    switch (dtype) {
        case F32:
            dsc_internal_rfftfreq(out, n, (f32) d);
            break;
        case F64:
            dsc_internal_rfftfreq(out, n, (f64) d);
            break;
        DSC_INVALID_CASE("dtype must be real");
    }

    return out;
}