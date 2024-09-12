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
static void assign_op(dsc_tensor *DSC_RESTRICT x,
                      const T start, const T step) noexcept {
    DSC_TENSOR_DATA(T, x);

    T val = start;
    dsc_for(i, x) {
        x_data[i] = val;
        val = add_op()(val, step);
    }
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
static void copy_op(const dsc_tensor *DSC_RESTRICT x,
                    dsc_tensor *DSC_RESTRICT out) noexcept {
    DSC_TENSOR_DATA(Tx, x);
    DSC_TENSOR_DATA(To, out);

    // Todo: I can probably do better but (if it works) it's fine for now
    dsc_for(i, out) {
        out_data[i] = cast_op().template operator()<Tx, To>(x_data[i]);
    }
}

template<typename Tx>
static void copy_op(const dsc_tensor *DSC_RESTRICT x,
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

#define DTYPE_DISPATCH(dtype, op, xa, xb) \
    switch (dtype)                        \
        case dsc_dtype::F32:              \
            op<f32>(xa, xb);              \
            break;                        \

static void copy(const dsc_tensor *DSC_RESTRICT x,
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
    dsc_tensor *res = dsc_sum(ctx, x, out, axis, keep_dims);

    const int axis_idx = dsc_tensor_dim(x, axis);
    const int axis_n = x->shape[axis_idx];

    switch (res->dtype) {
        case F32: {
            const f32 scale = 1.f / (f32) axis_n;
            res = dsc_mulc_f32(ctx, res, scale, res);
            break;
        }
        case F64: {
            const f64 scale = 1. / (f64) axis_n;
            res = dsc_mulc_f64(ctx, res, scale, res);
            break;
        }
        case C32: {
            const c32 scale = dsc_complex(c32, 1.f / (f32) axis_n, 0.f);
            res = dsc_mulc_c32(ctx, res, scale, res);
            break;
        }
        case C64: {
            const c64 scale = dsc_complex(c64, 1. / (f64) axis_n, 0.);
            res = dsc_mulc_c64(ctx, res, scale, res);
            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", res->dtype);
    }

    return res;
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