// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#include "dsc.h"
#include "cpu/dsc_cpu.h"
#include "dsc_device.h"
#include "dsc_tracing.h"
#include <cstdarg> // va_xxx
#include <cstring> // memset, memcpy


#define cast_binary_params()                 \
    const dsc_tensor *xa__ = xa, *xb__ = xb; \
    xa = dsc_cast(ctx, xa, out_dtype);       \
    xb = dsc_cast(ctx, xb, out_dtype)

#define cast_unary_params()         \
    const dsc_tensor *x__ = x;      \
    x = dsc_cast(ctx, x, out_dtype) \


// This needs to be a macro otherwise the pointer assignments to out, xa and xb
// would not work unless I pass them as pointers to pointers which is very ugly.
#define validate_binary_params(bool_as_float, int_as_float)                                    \
    DSC_ASSERT(xa != nullptr);                                                                 \
    DSC_ASSERT(xb != nullptr);                                                                 \
    DSC_ASSERT(can_broadcast(xa, xb));                                                         \
    DSC_ASSERT(xa->device == xb->device);                                                      \
                                                                                               \
    const int n_dim = DSC_MAX(xa->n_dim, xb->n_dim);                                           \
                                                                                               \
    int shape[DSC_MAX_DIMS];                                                                   \
    for (int i = 0; i < DSC_MAX_DIMS; ++i) shape[i] = DSC_MAX(xa->shape[i], xb->shape[i]);     \
                                                                                               \
    dsc_dtype out_dtype = DSC_DTYPE_CONVERSION_TABLE[xa->dtype][xb->dtype];                    \
    if ((bool_as_float && out_dtype == BOOL) || (int_as_float && out_dtype == I32)) {          \
        out_dtype = DSC_TYPE_AT_LEAST_FLOAT_TABLE[out_dtype];                                  \
    }                                                                                          \
    if (out == nullptr) {                                                                      \
        out = dsc_new_tensor(ctx, n_dim, &shape[DSC_MAX_DIMS - n_dim], out_dtype, xa->device); \
    } else {                                                                                   \
        DSC_ASSERT(out->dtype == out_dtype);                                                   \
        DSC_ASSERT(out->n_dim == n_dim);                                                       \
        DSC_ASSERT(memcmp(out->shape, shape, DSC_MAX_DIMS * sizeof(*shape)) == 0);             \
        DSC_ASSERT(out->device == xa->device);                                                 \
    }                                                                                          \
    cast_binary_params()

#define cleanup_binary_params()                   \
    do {                                          \
        if (xa__ != xa) dsc_tensor_free(ctx, xa); \
        if (xb__ != xb) dsc_tensor_free(ctx, xb); \
    } while (0)

#define validate_unary_params()                                                               \
    DSC_ASSERT(x != nullptr);                                                                 \
                                                                                              \
    const dsc_dtype out_dtype = DSC_TYPE_AT_LEAST_FLOAT_TABLE[x->dtype];                      \
                                                                                              \
    if (out == nullptr) {                                                                     \
        out = dsc_new_tensor(ctx, x->n_dim, &dsc_tensor_get_dim(x, 0), out_dtype, x->device); \
    } else {                                                                                  \
        DSC_ASSERT(out->dtype == out_dtype);                                                  \
        DSC_ASSERT(out->n_dim == x->n_dim);                                                   \
        DSC_ASSERT(out->device == x->device);                                                 \
        DSC_ASSERT(memcmp(out->shape, x->shape, DSC_MAX_DIMS * sizeof(out->shape[0])) == 0);  \
    }                                                                                         \
    cast_unary_params()

#define cleanup_unary_params()                 \
    do {                                       \
        if (x__ != x) dsc_tensor_free(ctx, x); \
    } while (0)

#define validate_reduce_params()                                                                        \
    DSC_ASSERT(x != nullptr);                                                                           \
                                                                                                        \
    const dsc_dtype out_dtype = DSC_TYPE_AT_LEAST_FLOAT_TABLE[x->dtype];                                \
                                                                                                        \
    const int axis_idx = dsc_tensor_dim_idx(x, axis);                                                   \
    DSC_ASSERT(axis_idx < DSC_MAX_DIMS);                                                                \
                                                                                                        \
    int out_shape[DSC_MAX_DIMS];                                                                        \
    int out_ndim = x->n_dim;                                                                            \
    if (keep_dims) {                                                                                    \
        memcpy(out_shape, x->shape, DSC_MAX_DIMS * sizeof(*out_shape));                                 \
        out_shape[axis_idx] = 1;                                                                        \
    } else {                                                                                            \
        out_ndim--;                                                                                     \
        const int out_offset = DSC_MAX_DIMS - out_ndim;                                                 \
        memset(out_shape, 1, out_offset * sizeof(*out_shape));                                          \
        for (int x_idx = DSC_MAX_DIMS - x->n_dim, out_idx = 0; x_idx < DSC_MAX_DIMS; ++x_idx) {         \
            if (x_idx == axis_idx)                                                                      \
                continue;                                                                               \
                                                                                                        \
            out_shape[out_offset + out_idx] = x->shape[x_idx];                                          \
            out_idx++;                                                                                  \
        }                                                                                               \
    }                                                                                                   \
                                                                                                        \
    if (out == nullptr) {                                                                               \
        out = dsc_new_tensor(ctx, out_ndim, &out_shape[DSC_MAX_DIMS - out_ndim], out_dtype, x->device); \
    } else {                                                                                            \
        DSC_ASSERT(out->dtype == out_dtype);                                                            \
        DSC_ASSERT(out->n_dim == out_ndim);                                                             \
        DSC_ASSERT(memcmp(out->shape, out_shape, DSC_MAX_DIMS * sizeof(*out_shape)) == 0);              \
    }                                                                                                   \
    cast_unary_params()


// If DEV is DEFAULT use the system default setting otherwise use the specified device
#define DSC_GET_DEV_ID(CTX, DEV) (DEV) == DEFAULT ? (CTX)->default_device : (DEV)
#define DSC_GET_DEVICE(CTX, DEV)                                                      \
    dsc_device *dev = (CTX)->devices[(CTX)->device_lookup[DSC_GET_DEV_ID(CTX, DEV)]]; \
    do {                                                                              \
        if (!dev)                                                                     \
            DSC_LOG_FATAL("device %d is null", (CTX)->default_device);                \
    } while (0)

#define DSC_DISPATCH(device, func, ...)                                      \
    do {                                                                     \
        const dsc_device_type dev_id = DSC_GET_DEV_ID(ctx, device);          \
        DSC_GET_DEVICE(ctx, device);                                         \
        if (dev_id == CPU)                                                   \
            dsc_cpu_##func(dev, ##__VA_ARGS__);                              \
        else                                                                 \
            DSC_LOG_FATAL("cannot dispatch to unknown device %d", (device)); \
    } while (0)

#define dsc_tensor_invalid(PTR)     (PTR)->ne <= 0
#define dsc_tensor_set_invalid(PTR) (PTR)->ne = -1

struct dsc_ctx {
    dsc_device *devices[DSC_MAX_DEVICES];
    int device_lookup[DSC_MAX_DEVICES];
    dsc_tensor *tensors;
    dsc_device_type default_device;
};

// ============================================================
// Initialization

dsc_ctx *dsc_ctx_init(const usize mem_size) {
    DSC_ASSERT(mem_size > 0);

    dsc_ctx *ctx = (dsc_ctx *) calloc(1, sizeof(dsc_ctx));
    DSC_ASSERT(ctx != nullptr);

    ctx->default_device = DSC_DEFAULT_DEVICE;

    ctx->devices[0] = dsc_cpu_device(mem_size);
    ctx->device_lookup[CPU] = 0;
    // Pre-allocate the tensor headers on the heap, this way we don't commit all the
    // memory upfront.
    ctx->tensors = (dsc_tensor *) calloc(DSC_MAX_OBJS, sizeof(dsc_tensor));

    dsc_internal_init_traces(DSC_MAX_TRACES);

    return ctx;
}

// ============================================================
// Cleanup/Teardown

void dsc_ctx_free(dsc_ctx *ctx) {
    for (int i = 0; i < DSC_MAX_DEVICES; ++i) {
        if (dsc_device *dev = ctx->devices[i]; dev) {
            dev->dispose(dev);
        }
    }

    dsc_internal_free_traces();

    free(ctx->tensors);
    free(ctx);
}

void dsc_tensor_free(dsc_ctx *ctx, dsc_tensor *x) {
    if (x == nullptr) return;
    // DSC_TRACE_TENSOR_FREE(x);

    DSC_GET_DEVICE(ctx, x->device);

    dsc_data_free(dev, x->buf);

    dsc_tensor_set_invalid(x);
}

// ============================================================
// Utilities

usize dsc_used_mem(dsc_ctx *ctx) {
    return ctx->devices[ctx->default_device]->used_mem;
}

void dsc_print_mem_usage(dsc_ctx *ctx) {
    printf("DSC mem usage:");
    for (int i = 0; i < DSC_MAX_DEVICES; ++i) {
        if (const dsc_device *dev = ctx->devices[i]; dev) {
            printf("\n %s: %ld/%ld (%.1f%%)",
                   DSC_DEVICE_NAMES[dev->type],
                   (usize) DSC_B_TO_MB(dev->used_mem),
                   (usize) DSC_B_TO_MB(dev->mem_size),
                   (f64) dev->used_mem / (f64) dev->mem_size * 1e2);
        }
    }
    printf("\n");
}

// ============================================================
// Tracing

void dsc_traces_record(dsc_ctx *, const bool record) {
    dsc_internal_record_traces(record);
}

void dsc_dump_traces(dsc_ctx *, const char *filename) {
    dsc_internal_dump_traces(filename);
}

void dsc_clear_traces(dsc_ctx *) {
    dsc_internal_clear_traces();
}

// ============================================================
// Tensor Creation

static DSC_INLINE dsc_tensor *find_empty_tensor(dsc_ctx *ctx) {
    for (int i = 0; i < DSC_MAX_OBJS; ++i) {
        if (dsc_tensor *x = &ctx->tensors[i]; dsc_tensor_invalid(x)) {
            return x;
        }
    }
    return nullptr;
}

dsc_tensor *dsc_new_tensor(dsc_ctx *ctx,
                           const int n_dim,
                           const int *shape,
                           const dsc_dtype dtype,
                           const dsc_device_type device,
                           dsc_data_buffer *buf,
                           const void *DSC_RESTRICT data,
                           const dsc_device_type data_device) {
    DSC_ASSERT((unsigned) n_dim <= DSC_MAX_DIMS);

    DSC_GET_DEVICE(ctx, device);

    // DSC_TRACE_TENSOR_NEW(shape, n_dim, dtype, backend);

    int ne = 1;
    for (int i = 0; i < n_dim; ++i) ne *= shape[i];

    DSC_ASSERT(ne > 0);
    dsc_tensor *new_tensor = find_empty_tensor(ctx);
    DSC_ASSERT(new_tensor != nullptr);

    if (buf == nullptr) {
        new_tensor->buf = dsc_data_alloc(dev, ne * DSC_DTYPE_SIZE[dtype]);
    } else {
        new_tensor->buf = buf;
        new_tensor->buf->refs++;
    }

    if (data != nullptr) {
        // Copy from data to this tensor buffer. Only support moving on device (cuda -> cuda) or from the cpu otherwise
        // we risk making a copy from cuda when cuda is not supported.
        DSC_ASSERT(data_device == CPU || data_device == device);
        DSC_DATA(void, new_tensor);
        dev->memcpy(new_tensor_data, data, ne * DSC_DTYPE_SIZE[dtype], data_device == device ? ON_DEVICE : TO_DEVICE);
    }

    new_tensor->dtype = dtype;
    new_tensor->ne = ne;
    new_tensor->n_dim = n_dim;
    new_tensor->device = DSC_GET_DEV_ID(ctx, device);

    // If n_dim is lower than DSC_MAX_DIM then we need to pre-fill the beginning of the array with 1
    for (int i = 0; i < DSC_MAX_DIMS; ++i) {
        new_tensor->shape[i] = i < (DSC_MAX_DIMS - n_dim) ? 1 : shape[i - (DSC_MAX_DIMS - n_dim)];
    }

    // Compute the stride
    memset(new_tensor->stride, 0, DSC_MAX_DIMS * sizeof(int));
    new_tensor->stride[DSC_MAX_DIMS - 1] = 1;
    for (int i = DSC_MAX_DIMS - 2; i >= 0; --i) {
        new_tensor->stride[i] = new_tensor->stride[i + 1] * new_tensor->shape[i + 1];
    }

    DSC_LOG_DEBUG("new tensor ptr=%p backend=%s n_dim=%d shape=[%d, %d, %d, %d] stride=[%d, %d, %d, %d] dtype=%s buffer=%p refs=%d",
                  new_tensor, DSC_DEVICE_NAMES[new_tensor->device], n_dim,
                  new_tensor->shape[0], new_tensor->shape[1], new_tensor->shape[2], new_tensor->shape[3],
                  new_tensor->stride[0], new_tensor->stride[1], new_tensor->stride[2], new_tensor->stride[3],
                  DSC_DTYPE_NAMES[dtype], new_tensor->buf, new_tensor->buf->refs
    );

    return new_tensor;
}

dsc_tensor *dsc_view(dsc_ctx *ctx, const dsc_tensor *x) {
    return dsc_new_view(ctx, x);
}

dsc_tensor *dsc_tensor_1d(dsc_ctx *ctx, const dsc_dtype dtype,
                          const int dim1, const dsc_device_type device,
                          const void *DSC_RESTRICT data,
                          const dsc_device_type data_device) {
    const int shape[DSC_MAX_DIMS] = {dim1};
    return dsc_new_tensor(ctx, 1, shape, dtype, device, nullptr, data, data_device);
}

dsc_tensor *dsc_tensor_2d(dsc_ctx *ctx, const dsc_dtype dtype,
                          const int dim1, const int dim2,
                          const dsc_device_type device,
                          const void *DSC_RESTRICT data,
                          const dsc_device_type data_device) {
    const int shape[DSC_MAX_DIMS] = {dim1, dim2};
    return dsc_new_tensor(ctx, 2, shape, dtype, device, nullptr, data, data_device);
}

dsc_tensor *dsc_tensor_3d(dsc_ctx *ctx, const dsc_dtype dtype,
                          const int dim1, const int dim2,
                          const int dim3,
                          const dsc_device_type device,
                          const void *DSC_RESTRICT data,
                          const dsc_device_type data_device) {
    const int shape[DSC_MAX_DIMS] = {dim1, dim2, dim3};
    return dsc_new_tensor(ctx, 3, shape, dtype, device, nullptr, data, data_device);
}

dsc_tensor *dsc_tensor_4d(dsc_ctx *ctx, const dsc_dtype dtype,
                          const int dim1, const int dim2,
                          const int dim3, const int dim4,
                          const dsc_device_type device,
                          const void *DSC_RESTRICT data,
                          const dsc_device_type data_device) {
    const int shape[DSC_MAX_DIMS] = {dim1, dim2, dim3, dim4};
    return dsc_new_tensor(ctx, 4, shape, dtype, device, nullptr, data, data_device);
}

template<typename T>
static DSC_INLINE dsc_tensor *wrap(dsc_ctx *ctx, const T val,
                                   const dsc_device_type device) {
    dsc_tensor *out = dsc_tensor_1d(ctx, dsc_type_mapping<T>::value, 1, device);

    DSC_GET_DEVICE(ctx, device);
    DSC_DATA(T, out);
    dev->memcpy(out_data, &val, sizeof(val), TO_DEVICE);
    return out;
}

dsc_tensor *dsc_wrap_bool(dsc_ctx *ctx, const bool val,
                         const dsc_device_type device) {
    return wrap(ctx, val, device);
}

dsc_tensor *dsc_wrap_i32(dsc_ctx *ctx, const i32 val,
                         const dsc_device_type device) {
    return wrap(ctx, val, device);
}

dsc_tensor *dsc_wrap_f32(dsc_ctx *ctx, const f32 val,
                         const dsc_device_type device) {
    return wrap(ctx, val, device);
}

dsc_tensor *dsc_wrap_f64(dsc_ctx *ctx, const f64 val,
                         const dsc_device_type device) {
    return wrap(ctx, val, device);
}

dsc_tensor *dsc_arange(dsc_ctx *ctx,
                       const int n,
                       const dsc_dtype dtype,
                       const dsc_device_type device) {
    // DSC_TRACE_ARANGE_OP(n, dtype);

    dsc_tensor *out = dsc_tensor_1d(ctx, dtype, n, device);
    DSC_DISPATCH(device, arange, out);
    return out;
}

dsc_tensor *dsc_randn(dsc_ctx *ctx,
                      const int n_dim,
                      const int *shape,
                      const dsc_dtype dtype,
                      const dsc_device_type device) {
    // DSC_TRACE_RANDN_OP(shape, n_dim, dtype);

    dsc_tensor *out = dsc_new_tensor(ctx, n_dim, shape, dtype, device);
    DSC_DISPATCH(device, randn, out);
    return out;
}

dsc_pair dsc_topk(dsc_ctx *ctx,
                  const dsc_tensor *DSC_RESTRICT x,
                  const int k, const int axis,
                  const bool largest) {
    // Return the top K largest (smallest) elements of x along the given axis
    DSC_ASSERT(x != nullptr);

    const int axis_idx = dsc_tensor_dim_idx(x, axis);
    const int axis_n = x->shape[axis_idx];
    DSC_ASSERT((unsigned) axis_idx < (unsigned) DSC_MAX_DIMS);
    DSC_ASSERT((unsigned) k < (unsigned) axis_n);

    int out_shape[DSC_MAX_DIMS]{};
    memcpy(out_shape, x->shape, DSC_MAX_DIMS * sizeof(*x->shape));
    out_shape[axis_idx] = k;

    // Allocate a temporary buffer, this will be used to sort the elements along the given axis
    dsc_tensor *tmp_values = dsc_tensor_1d(ctx, x->dtype, axis_n, x->device);
    dsc_tensor *tmp_indexes = dsc_tensor_1d(ctx, I32, axis_n, x->device);
    dsc_tensor *out_values = dsc_new_tensor(ctx, x->n_dim, &out_shape[dsc_tensor_dim_idx(x, 0)], x->dtype, x->device);
    dsc_tensor *out_indexes = dsc_new_tensor(ctx, x->n_dim, &out_shape[dsc_tensor_dim_idx(x, 0)], I32, x->device);

    DSC_DISPATCH(x->device, topk, x, tmp_values, tmp_indexes, out_values, out_indexes, k, axis_idx, largest);

    dsc_tensor_free(ctx, tmp_values);
    dsc_tensor_free(ctx, tmp_indexes);

    return {out_values, out_indexes};
}

dsc_tensor *dsc_multinomial(dsc_ctx *ctx,
                            const dsc_tensor *DSC_RESTRICT x,
                            const int num_samples) {
    DSC_ASSERT(x != nullptr);
    DSC_ASSERT(x->n_dim <= 2);
    DSC_ASSERT(x->dtype == F32 || x->dtype == F64);
    DSC_ASSERT((unsigned) num_samples <= (unsigned) dsc_tensor_get_dim(x, -1));

    int out_shape[DSC_MAX_DIMS]{};
    memcpy(out_shape, x->shape, DSC_MAX_DIMS * sizeof(*x->shape));
    out_shape[DSC_MAX_DIMS - 1] = num_samples;

    dsc_tensor *out = dsc_new_tensor(ctx, x->n_dim, &out_shape[dsc_tensor_dim_idx(x, 0)], I32, x->device);

    DSC_DISPATCH(x->device, multinomial, x, out, num_samples);

    return out;
}

dsc_tensor *dsc_cast(dsc_ctx *ctx, dsc_tensor *DSC_RESTRICT x,
                     const dsc_dtype new_dtype) {
    // DSC_TRACE_CAST_OP(x, new_dtype);

    if (x->dtype == new_dtype) return x;

    dsc_tensor *out = dsc_new_tensor(ctx, x->n_dim,
                                     &dsc_tensor_get_dim(x, 0),
                                     new_dtype, x->device);

    DSC_DISPATCH(x->device, cast, x, out);

    return out;
}

// ============================================================
// Tensor Manipulation

void dsc_copy(dsc_ctx *ctx,
              dsc_tensor *DSC_RESTRICT x,
              void *DSC_RESTRICT data,
              const usize nb,
              const dsc_device_type data_device) {
    DSC_ASSERT(x != nullptr);
    DSC_ASSERT(x->ne * DSC_DTYPE_SIZE[x->dtype] >= nb);
    DSC_ASSERT(x->device == data_device);

    DSC_DATA(void, x);
    DSC_GET_DEVICE(ctx, x->device);
    dev->memcpy(x_data, data, nb, ON_DEVICE);
}

dsc_tensor *dsc_reshape(dsc_ctx *ctx,
                        const dsc_tensor *DSC_RESTRICT x,
                        const int dimensions...) {
    DSC_ASSERT((unsigned) dimensions <= DSC_MAX_DIMS);

    int new_shape[DSC_MAX_DIMS];
    int new_ne = 1;
    int unknown_dim = -1;

    std::va_list args;
    va_start(args, dimensions);
    for (int i = 0; i < dimensions; ++i) {
        const int el = va_arg(args, int);

        if (el < 0) {
            if (unknown_dim == -1) unknown_dim = i;
            else DSC_LOG_FATAL("can only specify one unknown dim");
        } else {
            new_ne *= el;
            new_shape[i] = el;
        }

    }
    va_end(args);

    if (unknown_dim != -1) {
        if (x->ne % new_ne != 0) DSC_LOG_FATAL("cannot reshape %d into %d with an unknown dimension", x->ne, new_ne);

        new_shape[unknown_dim] = x->ne / new_ne;
        new_ne = x->ne;
    }

    // DSC_TRACE_RESHAPE_OP(x, dimensions, new_shape);

    DSC_ASSERT(x->ne == new_ne);

    return dsc_new_tensor(ctx, dimensions, new_shape, x->dtype, x->device, x->buf);
}

dsc_tensor *dsc_concat(dsc_ctx *ctx, const int axis,
                       const int tensors...) {
    DSC_ASSERT(tensors > 1);

    // DSC_TRACE_CONCAT_OP(tensors, axis);

    dsc_tensor **to_concat = (dsc_tensor **) alloca(tensors * sizeof(dsc_tensor *));
    std::va_list args;
    va_start(args, tensors);
    for (int i = 0; i < tensors; ++i) {
        dsc_tensor *el = va_arg(args, dsc_tensor *);
        DSC_ASSERT(el != nullptr);

        to_concat[i] = el;
    }
    va_end(args);

    // All the tensors must have the same dtype and the same number of dimensions and be on the same device
    const dsc_dtype dtype = to_concat[0]->dtype;
    const int n_dim = to_concat[0]->n_dim;
    const dsc_device_type device = to_concat[0]->device;
    for (int i = 1; i < tensors; ++i) {
        DSC_ASSERT(to_concat[i]->dtype == dtype);
        DSC_ASSERT(to_concat[i]->n_dim == n_dim);
        DSC_ASSERT(to_concat[i]->device == device);
    }

    DSC_GET_DEVICE(ctx, device);
    if (axis == DSC_VALUE_NONE) {
        // Flatten
        int ne = 0;
        for (int i = 0; i < tensors; ++i) ne += to_concat[i]->ne;

        dsc_tensor *out = dsc_tensor_1d(ctx, dtype, ne);
        usize offset = 0;
        for (int i = 0; i < tensors; ++i) {
            const dsc_tensor *src = to_concat[i];
            const usize nb = src->ne * DSC_DTYPE_SIZE[dtype];
            dev->memcpy((byte *) out->buf->data + offset, src->buf->data, nb, ON_DEVICE);
            offset += nb;
        }

        return out;
    }

    const int axis_idx = dsc_tensor_dim_idx(to_concat[0], axis);
    DSC_ASSERT(axis_idx < DSC_MAX_DIMS);

    int resulting_shape[DSC_MAX_DIMS];
    memcpy(resulting_shape, to_concat[0]->shape, DSC_MAX_DIMS * sizeof(*resulting_shape));

    // All the tensors must have the same shape expect for the axis dimension
    for (int i = 1; i < tensors; ++i) {
        for (int idx = 0; idx < DSC_MAX_DIMS; ++idx) {
            if (idx == axis_idx) {
                resulting_shape[axis_idx] += to_concat[i]->shape[idx];
                continue;
            }

            DSC_ASSERT(to_concat[i]->shape[idx] == to_concat[0]->shape[idx]);
        }
    }

    dsc_tensor *out = dsc_new_tensor(ctx, n_dim,
                                     &resulting_shape[dsc_tensor_dim_idx(to_concat[0], 0)],
                                     dtype, device);

    DSC_DISPATCH(out->device, concat, to_concat, tensors, out, axis_idx);

    return out;
}

dsc_tensor *dsc_transpose(dsc_ctx *ctx,
                          const dsc_tensor *DSC_RESTRICT x,
                          const int axes...) {
    // Transpose the given axes of tensor x.
    // If axes are not given (0) reverse the order of the axes of x.
    DSC_ASSERT(x != nullptr);

    if (x->n_dim == 1) {
        // Return a view of the same vector since a transpose is a NOP in this case
        return dsc_new_view(ctx, x);
    }

    int swap_axes[DSC_MAX_DIMS];
    if (axes == 0) {
        // [0, 1, .., N-1] --> [N-1, .., 1, 0]
        for (int i = 0; i < x->n_dim; ++i) swap_axes[i] = x->n_dim - (i + 1);
    } else {
        DSC_ASSERT(axes == x->n_dim);
        std::va_list args;
        va_start(args, axes);
        for (int i = 0; i < axes; ++i) {
            // Accept also negative axes
            swap_axes[i] = va_arg(args, int);
        }
        va_end(args);
    }
    // DSC_TRACE_TRANSPOSE_OP(x, swap_axes);

    int swapped_shape[DSC_MAX_DIMS], swapped_stride[DSC_MAX_DIMS];
    memcpy(swapped_shape, x->shape, DSC_MAX_DIMS * sizeof(*x->shape));
    memcpy(swapped_stride, x->stride, DSC_MAX_DIMS * sizeof(*x->stride));

    for (int i = 0; i < x->n_dim; ++i) {
        const int idx = dsc_tensor_dim_idx(x, swap_axes[i]);
        DSC_ASSERT((unsigned) idx < DSC_MAX_DIMS);

        swapped_shape[dsc_tensor_dim_idx(x, i)] = x->shape[idx];
        swapped_stride[dsc_tensor_dim_idx(x, i)] = x->stride[idx];
    }

    dsc_tensor *out = dsc_new_tensor(ctx, x->n_dim,
                                     &swapped_shape[dsc_tensor_dim_idx(x, 0)],
                                     x->dtype, x->device);

    DSC_DISPATCH(x->device, transpose, x, out, swapped_shape, swapped_stride);

    return out;
}

dsc_tensor *dsc_tril(dsc_ctx *ctx,
                     const dsc_tensor *DSC_RESTRICT x,
                     const int diagonal,
                     dsc_tensor *DSC_RESTRICT out) {
    DSC_ASSERT(x != nullptr);
    DSC_ASSERT(x->n_dim >= 2);

    if (out == nullptr) {
        out = dsc_new_like(ctx, x);
    } else {
        DSC_ASSERT(out->device == x->device);
        DSC_ASSERT(out->n_dim == x->n_dim);
        DSC_ASSERT(out->dtype == x->dtype);
        DSC_ASSERT(memcmp(out->shape, x->shape, DSC_MAX_DIMS * sizeof(*x->shape)) == 0);
    }

    DSC_DISPATCH(x->device, tril, x, diagonal, out);

    return out;
}

// ============================================================
// Indexing and Slicing
//

dsc_tensor *dsc_tensor_get_idx(dsc_ctx *ctx,
                               const dsc_tensor *DSC_RESTRICT x,
                               const int indexes...) {
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
        const int x_dim_i = dsc_tensor_get_dim(x, i);
        // Negative indexes mean accessing from the end
        if (idx < 0) idx += x_dim_i;

        DSC_ASSERT((unsigned) idx < (unsigned) x_dim_i);

        el_idx[i] = idx;
    }
    va_end(args);

    // DSC_TRACE_GET_IDX(x, el_idx, indexes);

    // Since we are wrapping scalars the resulting tensor will be always at least 1D
    const int out_n_dim = x->n_dim == indexes ? 1 : x->n_dim - indexes;
    // If we are indexing a single element then of course the output shape will be just 1
    int out_shape[DSC_MAX_DIMS] = {1};
    if (x->n_dim > indexes) {
        memcpy(out_shape, &x->shape[DSC_MAX_DIMS - out_n_dim], out_n_dim * sizeof(*x->shape));
    }

    dsc_tensor *out = dsc_new_tensor(ctx, out_n_dim, out_shape, x->dtype, x->device);

    int offset = 0;
    for (int i = 0; i < indexes; ++i) {
        offset += (dsc_tensor_get_stride(x, i) * el_idx[i]);
    }

    DSC_GET_DEVICE(ctx, x->device);

    DSC_DATA(void, out);
    DSC_DATA(byte, x);

    dev->memcpy(out_data, x_data + (offset * DSC_DTYPE_SIZE[x->dtype]),
                out->ne * DSC_DTYPE_SIZE[out->dtype], ON_DEVICE);

    return out;
}

static DSC_INLINE bool parse_slices(const dsc_tensor *DSC_RESTRICT x,
                                    dsc_slice *parsed_slices,
                                    bool *collapse_dim,
                                    const int slices,
                                    std::va_list args) {
    bool whole = true;

    for (int i = 0; i < slices; ++i) {
        dsc_slice slice = va_arg(args, dsc_slice);
        const int x_dim_i = dsc_tensor_get_dim(x, i);

        // The convention is to set all fields in the slice to the same value != NONE to signal
        // access to a single index rather than a slice (happens in mixed scenarios like x[:, 1])
        if (slice.start == slice.stop &&
            slice.start == slice.step &&
            slice.start != DSC_VALUE_NONE) {
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

        // If all the slices are in the form (:, :, :) the operation will be a simple 'whole tensor' operation
        // (either a copy or a linear set/fill) that won't require extra logic, just a simple for-loop.
        whole &= (slice.start == DSC_VALUE_NONE && slice.stop == DSC_VALUE_NONE && slice.step == DSC_VALUE_NONE);

        // If a field is marked using DSC_VALUE_NONE then replace it with the 'default' behaviour.
        // The default behaviour is controlled by step (see: https://numpy.org/doc/stable/user/basics.indexing.html)
        if (slice.step == DSC_VALUE_NONE) slice.step = 1;
        if (slice.start == DSC_VALUE_NONE) {
            if (slice.step > 0) slice.start = 0;
            else slice.start = x_dim_i - 1;
        }
        if (slice.stop == DSC_VALUE_NONE) {
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
    return whole;
}

dsc_tensor *dsc_tensor_get_slice(dsc_ctx *ctx,
                                 const dsc_tensor *DSC_RESTRICT x,
                                 const int slices...) {
    DSC_ASSERT(x != nullptr);
    DSC_ASSERT((unsigned) slices <= DSC_MAX_DIMS);

    if (slices > x->n_dim) {
        DSC_LOG_FATAL("too many slices");
    }

    dsc_slice el_slices[DSC_MAX_DIMS];
    bool collapse_dim[DSC_MAX_DIMS] = {false};

    std::va_list args;
    va_start(args, slices);
    const bool whole = parse_slices(x, el_slices, collapse_dim, slices, args);
    va_end(args);

    // DSC_TRACE_GET_SLICE(x, el_slices, slices);

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
            out_shape[out_idx] = dsc_tensor_get_dim(x, i);
        }
        out_idx += 1;
    }

    dsc_tensor *out = dsc_new_tensor(ctx, out_n_dim, out_shape, x->dtype, x->device);

    DSC_DISPATCH(out->device, get_slice, x, out, slices, el_slices, whole);

    return out;
}

dsc_tensor *dsc_tensor_get_tensor(dsc_ctx *ctx,
                                  const dsc_tensor *DSC_RESTRICT x,
                                  const dsc_tensor *DSC_RESTRICT indexes) {
    DSC_ASSERT(x != nullptr);
    DSC_ASSERT(indexes != nullptr);
    DSC_ASSERT(indexes->dtype == I32);
    DSC_ASSERT(x->device == indexes->device);

    // Note: right now this implements the behaviour of torch.Embedding
    DSC_ASSERT(x->n_dim == 2);
    const int out_ndim = indexes->n_dim + 1;
    DSC_ASSERT(out_ndim <= DSC_MAX_DIMS);

    int out_shape[DSC_MAX_DIMS]{};
    for (int i = 0; i < indexes->n_dim; ++i) out_shape[i] = dsc_tensor_get_dim(indexes, i);
    out_shape[indexes->n_dim] = dsc_tensor_get_dim(x, -1);

    dsc_tensor *out = dsc_new_tensor(ctx, out_ndim, out_shape, x->dtype, x->device);
    DSC_DISPATCH(x->device, get_tensor, x, indexes, out);
    return out;
}

void dsc_tensor_set_idx(dsc_ctx *ctx,
                        dsc_tensor *DSC_RESTRICT xa,
                        const dsc_tensor *DSC_RESTRICT xb,
                        const int indexes...) {
    DSC_ASSERT(xa != nullptr);
    DSC_ASSERT(xb != nullptr);
    DSC_ASSERT((unsigned) indexes <= (unsigned) xa->n_dim);
    DSC_ASSERT(xa->dtype == xb->dtype);
    DSC_ASSERT(xa->device == xb->device);

    // Use slices so it's easier to iterate
    dsc_slice el_slices[DSC_MAX_DIMS];

    std::va_list args;
    va_start(args, indexes);
    for (int i = 0; i < indexes; ++i) {
        const int idx = va_arg(args, int);
        const int x_dim_i = dsc_tensor_get_dim(xa, i);

        el_slices[i].start = idx;
        el_slices[i].stop = idx + 1;
        el_slices[i].step = 1;
        if (idx < 0) {
            el_slices[i].start += x_dim_i;
            el_slices[i].stop += x_dim_i;
        }
    }
    va_end(args);

    // DSC_TRACE_SET_IDX(xa, xb, el_slices, indexes);

    // If we do something like xa[2] and xa has more than one dimension then, the remaining
    // dimensions of xa and xb must be broadcastable together
    int xa_sub_shape[DSC_MAX_DIMS];
    for (int i = indexes; i < xa->n_dim; ++i)
        xa_sub_shape[i - indexes] = dsc_tensor_get_dim(xa, i - indexes);

    const bool xb_scalar = dsc_is_scalar(xb);
    const int xa_sub_ndim = xa->n_dim - indexes;

    if (xa_sub_ndim == 0) DSC_ASSERT(xb_scalar);

    if (!xb_scalar) {
        // If xb is not a scalar then its shape must be compatible with xa_sub_shape
        DSC_ASSERT(xb->n_dim == xa_sub_ndim);
        for (int i = 0; i < xa_sub_ndim; ++i) DSC_ASSERT(xa_sub_shape[i] == dsc_tensor_get_dim(xb, i));
    }

    DSC_DISPATCH(xa->device, set_slice, xa, xa_sub_ndim == 0, xb, xb_scalar, indexes, el_slices, false);
}

void dsc_tensor_set_slice(dsc_ctx *ctx,
                          dsc_tensor *DSC_RESTRICT xa,
                          const dsc_tensor *DSC_RESTRICT xb,
                          const int slices...) {
    DSC_ASSERT(xa != nullptr);
    DSC_ASSERT(xb != nullptr);
    DSC_ASSERT((unsigned) slices <= (unsigned) xa->n_dim);
    DSC_ASSERT(xa->dtype == xb->dtype);
    DSC_ASSERT(xa->device == xb->device);

    dsc_slice el_slices[DSC_MAX_DIMS];

    std::va_list args;
    va_start(args, slices);
    const bool whole = parse_slices(xa, el_slices, nullptr, slices, args);
    va_end(args);

    // DSC_TRACE_SET_SLICE(xa, xb, el_slices, slices);

    int xa_slice_shape[DSC_MAX_DIMS];
    for (int i = 0; i < xa->n_dim; ++i) {
        if (i < slices) {
            const dsc_slice slice_i = el_slices[i];
            const int ne_i = abs(slice_i.stop - slice_i.start);
            const int abs_step = abs(slice_i.step);
            xa_slice_shape[i] = (ne_i + abs_step - 1) / abs_step;
        } else {
            xa_slice_shape[i] = dsc_tensor_get_dim(xa, i);
        }
    }
    const bool xb_scalar = dsc_is_scalar(xb);
    if (!xb_scalar) {
        // Check whether xb is broadcastable with xa
        const int dims_to_compare = DSC_MIN(xa->n_dim, xb->n_dim);
        for (int i = 0; i < dims_to_compare; ++i) {
            const int xb_dim_i = dsc_tensor_get_dim(xb, i);
            const int xa_slice_i = xa_slice_shape[i];
            DSC_ASSERT(xa_slice_i == 1 || xb_dim_i == 1 || xa_slice_i == xb_dim_i);
        }
    }

    bool xa_scalar = true;
    for (int i = 0; i < xa->n_dim && xa_scalar; ++i)
        xa_scalar &= xa_slice_shape[i] == 1;

    DSC_DISPATCH(xa->device, set_slice, xa, xa_scalar, xb, xb_scalar, slices, el_slices, whole);
}

// ============================================================
// Binary Operations

static bool DSC_INLINE DSC_PURE can_broadcast(const dsc_tensor *DSC_RESTRICT xa,
                                              const dsc_tensor *DSC_RESTRICT xb) {
    bool can_broadcast = true;
    for (int i = 0; i < DSC_MAX_DIMS && can_broadcast; ++i) {
        can_broadcast = xa->shape[i] == xb->shape[i] ||
                        xa->shape[i] == 1 ||
                        xb->shape[i] == 1;
    }

    return can_broadcast;
}

dsc_tensor *dsc_add(dsc_ctx *ctx,
                    dsc_tensor *xa,
                    dsc_tensor *xb,
                    dsc_tensor *out) {
    // DSC_TRACE_BINARY_OP(xa, xb, out);

    validate_binary_params(false, false);

    DSC_DISPATCH(xa->device, add, xa, xb, out);

    cleanup_binary_params();

    return out;
}

dsc_tensor *dsc_sub(dsc_ctx *ctx,
                    dsc_tensor *xa,
                    dsc_tensor *xb,
                    dsc_tensor *out) {
    // DSC_TRACE_BINARY_OP(xa, xb, out);

    validate_binary_params(false, false);

    DSC_DISPATCH(xa->device, sub, xa, xb, out);

    cleanup_binary_params();

    return out;
}

dsc_tensor *dsc_mul(dsc_ctx *ctx,
                    dsc_tensor *xa,
                    dsc_tensor *xb,
                    dsc_tensor *out) {
    // DSC_TRACE_BINARY_OP(xa, xb, out);

    validate_binary_params(false, false);

    DSC_DISPATCH(xa->device, mul, xa, xb, out);

    cleanup_binary_params();

    return out;
}

dsc_tensor *dsc_div(dsc_ctx *ctx,
                    dsc_tensor *xa,
                    dsc_tensor *xb,
                    dsc_tensor *out) {
    // DSC_TRACE_BINARY_OP(xa, xb, out);

    validate_binary_params(true, true);

    DSC_DISPATCH(xa->device, div, xa, xb, out);

    cleanup_binary_params();

    return out;
}

dsc_tensor *dsc_pow(dsc_ctx *ctx,
                    dsc_tensor *xa,
                    dsc_tensor *xb,
                    dsc_tensor *out) {
    // DSC_TRACE_BINARY_OP(xa, xb, out);

    validate_binary_params(true, false);

    DSC_DISPATCH(xa->device, pow, xa, xb, out);

    cleanup_binary_params();

    return out;
}

dsc_tensor *dsc_matmul(dsc_ctx *ctx,
                       dsc_tensor *DSC_RESTRICT xa,
                       dsc_tensor *DSC_RESTRICT xb,
                       dsc_tensor *DSC_RESTRICT out) {
    // Notes: it would make sense to add also a 'transpose' flag for xa and xb
    //        as well as an 'alpha' coefficient (so we end up performing out = alpha * (xa @ xb))
    DSC_ASSERT(xa->device == xb->device);

    if (xa->n_dim < 2 || xb->n_dim < 2) return dsc_mul(ctx, xa, xb, out);

    const int xa_rows = dsc_tensor_get_dim(xa, -2);
    const int xa_cols = dsc_tensor_get_dim(xa, -1);
    const int xb_rows = dsc_tensor_get_dim(xb, -2);
    const int xb_cols = dsc_tensor_get_dim(xb, -1);
    if (xa_cols != xb_rows) DSC_LOG_FATAL("can't multiply (%d, %d) by (%d, %d)",
                                          xa_rows, xa_cols, xb_rows, xb_cols);

    bool can_broadcast = true;
    for (int i = 0; i < DSC_MAX_DIMS - 2 && can_broadcast; ++i) {
        if  (xa->shape[i] != xb->shape[i] && xa->shape[i] != 1 && xb->shape[i] != 1) can_broadcast = false;
    }
    DSC_ASSERT(can_broadcast);

    int out_shape[DSC_MAX_DIMS];
    for (int i = 0; i < DSC_MAX_DIMS - 2; ++i) out_shape[i] = DSC_MAX(xa->shape[i], xb->shape[i]);

    out_shape[DSC_MAX_DIMS - 2] = xa_rows;
    out_shape[DSC_MAX_DIMS - 1] = xb_cols;
    const int out_ndim = DSC_MAX(xa->n_dim, xb->n_dim);

    const dsc_dtype out_dtype = DSC_TYPE_AT_LEAST_FLOAT_TABLE[DSC_DTYPE_CONVERSION_TABLE[xa->dtype][xb->dtype]];

    if (out != nullptr) {
        DSC_ASSERT(memcmp(out_shape, out->shape, DSC_MAX_DIMS * sizeof(*out->shape)) == 0);
        DSC_ASSERT(out_ndim == out->n_dim);
        DSC_ASSERT(out->dtype == out_dtype);
        DSC_ASSERT(out->device == xa->device);
    } else {
        out = dsc_new_tensor(ctx, out_ndim, &out_shape[DSC_MAX_DIMS - out_ndim], out_dtype, xa->device);
    }

    cast_binary_params();

    // Matmul requires the result to be zero-initialized
    DSC_GET_DEVICE(ctx, xa->device);
    DSC_DATA(void, out);
    dev->memset(out_data, 0, out->ne * DSC_DTYPE_SIZE[out->dtype]);

    DSC_DISPATCH(xa->device, matmul, xa, xb, out);

    cleanup_binary_params();

    return out;
}

dsc_tensor *dsc_compare(dsc_ctx *ctx,
                        const dsc_tensor *xa,
                        const dsc_tensor *xb,
                        const dsc_comparison_op comp,
                        dsc_tensor *out) {
    // Todo: mostly duplicated code from `validate_binary_params` but without the casting
    DSC_ASSERT(xa != nullptr);                                                                 
    DSC_ASSERT(xb != nullptr);                                                                 
    DSC_ASSERT(can_broadcast(xa, xb));                                                         
    DSC_ASSERT(xa->device == xb->device);                                                      
                                                                                               
    const int n_dim = DSC_MAX(xa->n_dim, xb->n_dim);                                           
                                                                                               
    int shape[DSC_MAX_DIMS];                                                                   
    for (int i = 0; i < DSC_MAX_DIMS; ++i) shape[i] = DSC_MAX(xa->shape[i], xb->shape[i]);     
                                                                                               
    if (out == nullptr) {
        out = dsc_new_tensor(ctx, n_dim, &shape[DSC_MAX_DIMS - n_dim], BOOL, xa->device);
    } else {                                                                                   
        DSC_ASSERT(out->dtype == BOOL);
        DSC_ASSERT(out->n_dim == n_dim);                                                       
        DSC_ASSERT(memcmp(out->shape, shape, DSC_MAX_DIMS * sizeof(*shape)) == 0);             
        DSC_ASSERT(out->device == xa->device);                                                 
    }

    DSC_DISPATCH(xa->device, compare, xa, xb, comp, out);

    return out;
}

void dsc_masked_fill(dsc_ctx *ctx,
                     dsc_tensor *x,
                     const dsc_tensor *mask,
                     const f64 value) {
    DSC_ASSERT(x != nullptr);
    DSC_ASSERT(mask != nullptr);
    DSC_ASSERT(x->device == mask->device);
    DSC_ASSERT(mask->dtype == BOOL);
    DSC_ASSERT(x->dtype == F32 || x->dtype == F64);

    // Mask must be broadcastable with the shape of x, not the other way around
    DSC_ASSERT(x->n_dim >= mask->n_dim);
    DSC_ASSERT(can_broadcast(x, mask));
    for (int i = 0; i < DSC_MAX_DIMS; ++i) {
        DSC_ASSERT(x->shape[i] >= mask->shape[i]);
    }

    DSC_DISPATCH(x->device, masked_fill, x, mask, value);
}

// ============================================================
// Unary Operations

dsc_tensor *dsc_cos(dsc_ctx *ctx,
                    dsc_tensor *DSC_RESTRICT x,
                    dsc_tensor *DSC_RESTRICT out) {
    // DSC_TRACE_UNARY_OP(x, out);

    validate_unary_params();

    DSC_DISPATCH(x->device, cos, x, out);

    cleanup_unary_params();

    return out;
}

dsc_tensor *dsc_sin(dsc_ctx *ctx,
                    dsc_tensor *DSC_RESTRICT x,
                    dsc_tensor *DSC_RESTRICT out) {
    // DSC_TRACE_UNARY_OP(x, out);

    validate_unary_params();

    DSC_DISPATCH(x->device, sin, x, out);

    cleanup_unary_params();

    return out;
}

dsc_tensor *dsc_tanh(dsc_ctx *ctx,
                     dsc_tensor *DSC_RESTRICT x,
                     dsc_tensor *DSC_RESTRICT out) {
    validate_unary_params();

    DSC_DISPATCH(x->device, tanh, x, out);

    cleanup_unary_params();

    return out;
}

dsc_tensor *dsc_exp(dsc_ctx *ctx,
                    dsc_tensor *DSC_RESTRICT x,
                    dsc_tensor *DSC_RESTRICT out) {
    // DSC_TRACE_UNARY_OP(x, out);

    validate_unary_params();

    DSC_DISPATCH(x->device, exp, x, out);

    cleanup_unary_params();

    return out;
}

dsc_tensor *dsc_sqrt(dsc_ctx *ctx,
                     dsc_tensor *DSC_RESTRICT x,
                     dsc_tensor *DSC_RESTRICT out) {
    // DSC_TRACE_UNARY_OP(x, out);

    validate_unary_params();

    DSC_DISPATCH(x->device, sqrt, x, out);

    cleanup_unary_params();

    return out;
}

// ============================================================
// Unary Operations Along Axis


dsc_tensor *dsc_sum(dsc_ctx *ctx,
                    dsc_tensor *DSC_RESTRICT x,
                    dsc_tensor *DSC_RESTRICT out,
                    const int axis,
                    const bool keep_dims) {
    // DSC_TRACE_UNARY_AXIS_OP(x, out, axis, keep_dims);

    validate_reduce_params();

    DSC_DISPATCH(x->device, sum, x, out, axis_idx);

    cleanup_unary_params();

    return out;
}

dsc_tensor *dsc_max(dsc_ctx *ctx,
                     dsc_tensor *DSC_RESTRICT x,
                     dsc_tensor *DSC_RESTRICT out,
                     const int axis,
                     const bool keep_dims) {
    // DSC_TRACE_UNARY_AXIS_OP(x, out, axis, keep_dims);

    validate_reduce_params();

    DSC_DISPATCH(x->device, max, x, out, axis_idx);

    cleanup_unary_params();

    return out;
}

dsc_tensor *dsc_min(dsc_ctx *ctx,
                     dsc_tensor *DSC_RESTRICT x,
                     dsc_tensor *DSC_RESTRICT out,
                     const int axis,
                     const bool keep_dims) {
    // DSC_TRACE_UNARY_AXIS_OP(x, out, axis, keep_dims);

    validate_reduce_params();

    DSC_DISPATCH(x->device, min, x, out, axis_idx);

    cleanup_unary_params();

    return out;
}