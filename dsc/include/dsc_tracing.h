// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "dsc.h"

#if defined(DSC_ENABLE_TRACING)
#   include <cstring>
#   include <unistd.h>     // getpid()
#   include <pthread.h>    // pthread_self()


#define DSC_TRACE_NAME_MAX  ((int) 32)
#define DSC_TRACE_CAT_MAX   ((int) 16)

#define DSC_INSERT_TYPED_TRACE(T, cat_, type_) \
    dsc_trace_tracker<T> trace__{__FUNCTION__, (cat_), (type_), &args__}

#define DSC_TRACE_SET_TENSOR(X, field)  \
    args__.field.n_dim = (X)->n_dim;    \
    memcpy(args__.field.shape, &(X)->shape[dsc_tensor_dim((X), 0)], (X)->n_dim * sizeof(*(X)->shape));  \
    args__.field.dtype = (X)->dtype;        \
    args__.field.backend = (X)->backend;    \
    args__.field.addr = (uintptr_t) (X)    

#define DSC_TRACE_TENSOR_NEW(shape_, n_dim_, dtype_, backend_)  \
    dsc_tensor_alloc_args args__{};                             \
    memcpy(&args__.x.shape, (shape_), (n_dim_) * sizeof(*(shape_)));    \
    args__.x.n_dim = (n_dim_);      \
    args__.x.dtype = (dtype_);      \
    args__.x.backend = (backend_);  \
    args__.x.addr = 0;              \
    DSC_INSERT_TYPED_TRACE(dsc_tensor_alloc_args, "alloc", DSC_TENSOR_ALLOC)

#define DSC_TRACE_TENSOR_FREE(X)    \
    dsc_tensor_alloc_args args__{}; \
    DSC_TRACE_SET_TENSOR(X, x);     \
    DSC_INSERT_TYPED_TRACE(dsc_tensor_alloc_args, "free", DSC_TENSOR_FREE)

#define DSC_TRACE_BINARY_OP(XA, XB, OUT)    \
    dsc_binary_args args__{};               \
    DSC_TRACE_SET_TENSOR(XA, xa);           \
    DSC_TRACE_SET_TENSOR(XB, xb);           \
    if ((OUT) != nullptr) {                 \
        DSC_TRACE_SET_TENSOR(OUT, out);     \
    }                                       \
    args__.with_out = (OUT) != nullptr;     \
    DSC_INSERT_TYPED_TRACE(dsc_binary_args, "op;binary", DSC_BINARY_OP)

#define DSC_TRACE_UNARY_OP(X, OUT)      \
    dsc_unary_args args__{};            \
    DSC_TRACE_SET_TENSOR(X, x);         \
    if ((OUT) != nullptr) {             \
        DSC_TRACE_SET_TENSOR(OUT, out); \
    }                                   \
    args__.with_out = (OUT) != nullptr; \
    DSC_INSERT_TYPED_TRACE(dsc_unary_args, "op;unary", DSC_UNARY_OP)

#define DSC_TRACE_UNARY_NO_OUT_OP(X)    \
    dsc_unary_no_out_args args__{};     \
    DSC_TRACE_SET_TENSOR(X, x);         \
    DSC_INSERT_TYPED_TRACE(dsc_unary_no_out_args, "op;unary", DSC_UNARY_NO_OUT_OP)

#define DSC_TRACE_UNARY_AXIS_OP(X, OUT, axis_, keep_dims_)  \
    dsc_unary_axis_args args__{};       \
    DSC_TRACE_SET_TENSOR(X, x);         \
    if ((OUT) != nullptr) {             \
        DSC_TRACE_SET_TENSOR(OUT, out); \
    }                                   \
    args__.with_out = (OUT) != nullptr; \
    args__.axis = (axis_);              \
    args__.keep_dims = (keep_dims_);    \
    DSC_INSERT_TYPED_TRACE(dsc_unary_axis_args, "op;unary", DSC_UNARY_AXIS_OP)

#define DSC_TRACE_FFT_OP(X, OUT, n_, axis_, type_, fwd_)    \
    dsc_fft_args args__{};                  \
    args__.n = (n_);                        \
    args__.axis = (axis_);                  \
    args__.type = (type_);                  \
    args__.forward = (fwd_);                \
    DSC_TRACE_SET_TENSOR(X, x);             \
    if ((OUT) != nullptr) {                 \
        DSC_TRACE_SET_TENSOR(OUT, out);     \
    }                                       \
    args__.with_out = (OUT) != nullptr;     \
    DSC_INSERT_TYPED_TRACE(dsc_fft_args, "op;fft", DSC_FFT_OP)

#define DSC_TRACE_PLAN_FFT(n_, fft_n_, fft_type_, dtype_)               \
    dsc_plan_fft_args args__{(n_), (fft_n_), (fft_type_), (dtype_)};    \
    DSC_INSERT_TYPED_TRACE(dsc_plan_fft_args, "op;fft;plan", DSC_PLAN_FFT)

#define DSC_TRACE_GET_IDX(X, indexes_, n_indexes_)  \
    dsc_get_idx_args args__{};                      \
    DSC_TRACE_SET_TENSOR(X, x);                     \
    memcpy(args__.indexes, (indexes_), (n_indexes_) * sizeof(*(indexes_))); \
    args__.n_indexes = (n_indexes_);    \
    DSC_INSERT_TYPED_TRACE(dsc_get_idx_args, "idx;get", DSC_GET_IDX)

#define DSC_TRACE_GET_SLICE(X, slices_, n_slices_)  \
    dsc_get_slice_args args__{};                    \
    DSC_TRACE_SET_TENSOR(X, x);                     \
    memcpy(args__.slices, (slices_), (n_slices_) * sizeof(*(slices_))); \
    args__.n_slices = (n_slices_);  \
    DSC_INSERT_TYPED_TRACE(dsc_get_slice_args, "slice;get", DSC_GET_SLICE)

#define DSC_TRACE_SET_IDX(XA, XB, indexes_, n_indexes_) \
    dsc_set_idx_args args__{};                          \
    DSC_TRACE_SET_TENSOR(XA, xa);                       \
    DSC_TRACE_SET_TENSOR(XB, xb);                       \
    memcpy(args__.indexes, (indexes_), (n_indexes_) * sizeof(*(indexes_))); \
    args__.n_indexes = (n_indexes_);    \
    DSC_INSERT_TYPED_TRACE(dsc_set_idx_args, "idx;set", DSC_SET_IDX)

#define DSC_TRACE_SET_SLICE(XA, XB, slices_, n_slices_) \
    dsc_set_slice_args args__{};                        \
    DSC_TRACE_SET_TENSOR(XA, xa);                       \
    DSC_TRACE_SET_TENSOR(XB, xb);                       \
    memcpy(args__.slices, (slices_), (n_slices_) * sizeof(*(slices_))); \
    args__.n_slices = (n_slices_);  \
    DSC_INSERT_TYPED_TRACE(dsc_set_slice_args, "slice;set", DSC_SET_SLICE)

#define DSC_TRACE_CAST_OP(X, new_dtype_)    \
    dsc_cast_args args__{};                 \
    DSC_TRACE_SET_TENSOR(X, x);             \
    args__.new_dtype = (new_dtype_);        \
    DSC_INSERT_TYPED_TRACE(dsc_cast_args, "op;cast", DSC_CAST_OP)

#define DSC_TRACE_RANDN_OP(shape_, n_dim_, dtype_)  \
    dsc_randn_args args__{};                        \
    memcpy(&args__.shape, (shape_), (n_dim_) * sizeof(*(shape_))); \
    args__.n_dim = (n_dim_);    \
    args__.dtype = (dtype_);    \
    DSC_INSERT_TYPED_TRACE(dsc_randn_args, "op;randn", DSC_RANDN_OP)

#define DSC_TRACE_ARANGE_OP(n_, dtype_) \
    dsc_arange_args args__{};           \
    args__.n = (n_);                    \
    args__.dtype = (dtype_);            \
    DSC_INSERT_TYPED_TRACE(dsc_arange_args, "op;arange", DSC_ARANGE_OP)

#define DSC_TRACE_RESHAPE_OP(X, new_ndim_, new_shape_)  \
    dsc_reshape_args args__{};                          \
    DSC_TRACE_SET_TENSOR(X, x);                         \
    args__.new_ndim = (new_ndim_);                      \
    memcpy(args__.new_shape, (new_shape_), (new_ndim_) * sizeof(*(new_shape_))); \
    DSC_INSERT_TYPED_TRACE(dsc_reshape_args, "op;reshape", DSC_RESHAPE_OP)

#define DSC_TRACE_CONCAT_OP(tensors_, axis_)    \
    dsc_concat_args args__{};                   \
    args__.tensors = (tensors_);                \
    args__.axis_ = (axis_);                     \
    DSC_INSERT_TYPED_TRACE(dsc_concat_args, "op;concat", DSC_CONCAT_OP)

#define DSC_TRACE_TRANSPOSE_OP(X, swap_axes_)   \
    dsc_transpose_args args__{};                \
    DSC_TRACE_SET_TENSOR(X, x);                 \
    memcpy(args__.swap_axes, (swap_axes_), (X)->n_dim * sizeof(*(swap_axes_))); \
    DSC_INSERT_TYPED_TRACE(dsc_transpose_args, "op;transpose", DSC_TRANSPOSE_OP)


enum dsc_trace_type : u8 {
    DSC_TENSOR_ALLOC,
    DSC_TENSOR_FREE,
    DSC_UNARY_OP,
    DSC_UNARY_NO_OUT_OP,
    DSC_UNARY_AXIS_OP,
    DSC_BINARY_OP,
    DSC_FFT_OP,
    DSC_PLAN_FFT,
    DSC_GET_IDX,
    DSC_GET_SLICE,
    DSC_SET_IDX,
    DSC_SET_SLICE,
    DSC_CAST_OP,
    DSC_RANDN_OP,
    DSC_ARANGE_OP,
    DSC_RESHAPE_OP,
    DSC_CONCAT_OP,
    DSC_TRANSPOSE_OP,
};

struct dsc_tensor_args {
    int shape[DSC_MAX_DIMS];
    uintptr_t addr;
    int n_dim;
    dsc_backend_type backend;
    dsc_dtype dtype;
};

struct dsc_tensor_alloc_args {
    dsc_tensor_args x;
};

struct dsc_fft_args {
    dsc_tensor_args x, out;
    int n, axis;
    dsc_fft_type type;
    bool forward, with_out;
};

struct dsc_unary_args {
    dsc_tensor_args x, out;
    bool with_out;
};

struct dsc_unary_no_out_args {
    dsc_tensor_args x;
};

struct dsc_unary_axis_args {
    dsc_tensor_args x, out;
    int axis;
    bool keep_dims, with_out;
};

struct dsc_binary_args {
    dsc_tensor_args xa, xb, out;
    bool with_out;
};

struct dsc_plan_fft_args {
    int requested_n, fft_n;
    dsc_fft_type type;
    dsc_dtype dtype;
};

struct dsc_get_idx_args {
    dsc_tensor_args x;
    int indexes[DSC_MAX_DIMS];
    int n_indexes;
};

struct dsc_get_slice_args {
    dsc_tensor_args x;
    dsc_slice slices[DSC_MAX_DIMS];
    int n_slices;
};

struct dsc_set_idx_args {
    dsc_tensor_args xa, xb;
    dsc_slice indexes[DSC_MAX_DIMS];
    int n_indexes;
};

struct dsc_set_slice_args {
    dsc_tensor_args xa, xb;
    dsc_slice slices[DSC_MAX_DIMS];
    int n_slices;
};

struct dsc_cast_args {
    dsc_tensor_args x;
    dsc_dtype new_dtype;
};

struct dsc_randn_args {
    int shape[DSC_MAX_DIMS];
    int n_dim;
    dsc_dtype dtype;
};

struct dsc_arange_args {
    int n;
    dsc_dtype dtype;
};

struct dsc_reshape_args {
    dsc_tensor_args x;
    int new_shape[DSC_MAX_DIMS];
    int new_ndim;
};

struct dsc_concat_args {
    int tensors;
    int axis;
};

struct dsc_transpose_args {
    dsc_tensor_args x;
    int swap_axes[DSC_MAX_DIMS];
};

struct dsc_trace {
    char name[DSC_TRACE_NAME_MAX], cat[DSC_TRACE_CAT_MAX];
    u64 tid, ts; // Timestamp of the event in us
    int pid;
    char phase; // Phase of the event (B=begin, E=end, X=complete ecc...)
    dsc_trace_type type;
    union {
        dsc_tensor_alloc_args tensor_alloc;
        dsc_fft_args fft;
        dsc_plan_fft_args plan_fft;
        dsc_unary_args unary;
        dsc_unary_no_out_args unary_no_out;
        dsc_unary_axis_args unary_axis;
        dsc_binary_args binary;
        dsc_get_idx_args get_idx;
        dsc_get_slice_args get_slice;
        dsc_set_idx_args set_idx;
        dsc_set_slice_args set_slice;
        dsc_cast_args cast;        
        dsc_randn_args randn;
        dsc_arange_args arange;
        dsc_reshape_args reshape;
        dsc_concat_args concat;
        dsc_transpose_args transpose;
    };
};

struct dsc_trace_ctx {
    dsc_trace *traces;
    u64 n_traces, max_traces;
    bool record;
};

extern dsc_trace_ctx *g_trace_ctx;

static DSC_INLINE u64 dsc_time_us() noexcept {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (u64) (ts.tv_sec * 1'000'000ULL) + (u64) (ts.tv_nsec / 1'000ULL);
}

template <typename T>
struct dsc_trace_tracker {
    dsc_trace_tracker(const char *name, const char *cat,
                      const dsc_trace_type type, const T *data) noexcept :
            data_(data), name_(name), cat_(cat),
            tid_(pthread_self()), pid_(getpid()), type_(type)  {
        if (g_trace_ctx->record &&
            g_trace_ctx->n_traces < g_trace_ctx->max_traces) {
            dsc_trace *t = &g_trace_ctx->traces[g_trace_ctx->n_traces++];
            pre_fill(t, 'B');
            fill_data(t);
            t->ts = dsc_time_us();
        }
    }

    ~dsc_trace_tracker() noexcept {
        if (g_trace_ctx->record &&
            g_trace_ctx->n_traces < g_trace_ctx->max_traces) {
            dsc_trace *t = &g_trace_ctx->traces[g_trace_ctx->n_traces++];
            t->ts = dsc_time_us();
            pre_fill(t, 'E');
            fill_data(t);
        }
    }

private:
    DSC_INLINE void pre_fill(dsc_trace *t, const char phase) const noexcept {
        strncpy(t->name, name_, DSC_TRACE_NAME_MAX);
        strncpy(t->cat, cat_, DSC_TRACE_CAT_MAX);
        t->pid = pid_;
        t->tid = tid_;
        t->phase = phase;
        t->type = type_;
    }

    DSC_INLINE void fill_data(dsc_trace *t) const noexcept {
        // Todo: this can easily be replaced by a macro or something more concise
         if constexpr (dsc_is_type<T, dsc_tensor_alloc_args>()) {
            const dsc_tensor_alloc_args *args = (const dsc_tensor_alloc_args *) data_;
            memcpy(&t->tensor_alloc, args, sizeof(*args));
        } else if constexpr (dsc_is_type<T, dsc_binary_args>()) {
            const dsc_binary_args *args = (const dsc_binary_args *) data_;
            memcpy(&t->binary, args, sizeof(*args));
        } else if constexpr (dsc_is_type<T, dsc_unary_args>()) {
            const dsc_unary_args *args = (const dsc_unary_args *) data_;
            memcpy(&t->unary, args, sizeof(*args));
        } else if constexpr (dsc_is_type<T, dsc_unary_no_out_args>()) {
            const dsc_unary_no_out_args *args = (const dsc_unary_no_out_args *) data_;
            memcpy(&t->unary_no_out, args, sizeof(*args));
        } else if constexpr (dsc_is_type<T, dsc_unary_axis_args>()) {
            const dsc_unary_axis_args *args = (const dsc_unary_axis_args *) data_;
            memcpy(&t->unary_axis, args, sizeof(*args));
        } else if constexpr (dsc_is_type<T, dsc_plan_fft_args>()) {
            const dsc_plan_fft_args *args = (const dsc_plan_fft_args *) data_;
            memcpy(&t->plan_fft, args, sizeof(*args));
        } else if constexpr (dsc_is_type<T, dsc_fft_args>()) {
            const dsc_fft_args *args = (const dsc_fft_args *) data_;
            memcpy(&t->fft, args, sizeof(*args));
        } else if constexpr (dsc_is_type<T, dsc_get_idx_args>()) {
            const dsc_get_idx_args *args = (const dsc_get_idx_args *) data_;
            memcpy(&t->get_idx, args, sizeof(*args));
        } else if constexpr (dsc_is_type<T, dsc_get_slice_args>()) {
            const dsc_get_slice_args *args = (const dsc_get_slice_args *) data_;
            memcpy(&t->get_slice, args, sizeof(*args));
        } else if constexpr (dsc_is_type<T, dsc_set_idx_args>()) {
            const dsc_set_idx_args *args = (const dsc_set_idx_args *) data_;
            memcpy(&t->set_idx, args, sizeof(*args));
        } else if constexpr (dsc_is_type<T, dsc_set_slice_args>()) {
            const dsc_set_slice_args *args = (const dsc_set_slice_args *) data_;
            memcpy(&t->set_slice, args, sizeof(*args));
        } else if constexpr (dsc_is_type<T, dsc_cast_args>()) {
            const dsc_cast_args *args = (const dsc_cast_args *) data_;
            memcpy(&t->cast, args, sizeof(*args));
        } else if constexpr (dsc_is_type<T, dsc_randn_args>()) {
            const dsc_randn_args *args = (const dsc_randn_args *) data_;
            memcpy(&t->randn, args, sizeof(*args));
        } else if constexpr (dsc_is_type<T, dsc_arange_args>()) {
            const dsc_arange_args *args = (const dsc_arange_args *) data_;
            memcpy(&t->arange, args, sizeof(*args));
        } else if constexpr (dsc_is_type<T, dsc_reshape_args>()) {
            const dsc_reshape_args *args = (const dsc_reshape_args *) data_;
            memcpy(&t->reshape, args, sizeof(*args));
        } else if constexpr (dsc_is_type<T, dsc_concat_args>()) {
            const dsc_concat_args *args = (const dsc_concat_args *) data_;
            memcpy(&t->concat, args, sizeof(*args));
        } else if constexpr (dsc_is_type<T, dsc_transpose_args>()) {
            const dsc_transpose_args *args = (const dsc_transpose_args *) data_;
            memcpy(&t->transpose, args, sizeof(*args));
        } else {
            static_assert("T is not supported");
        }
    }

    const T *data_;
    const char *name_, *cat_;
    const u64 tid_;
    const int pid_;
    const dsc_trace_type type_;
};

#else

#define DSC_TRACE_TENSOR_NEW(shape_, n_dim_, dtype_, backend_)  ((void) 0)
#define DSC_TRACE_TENSOR_FREE(X)                                ((void) 0)
#define DSC_TRACE_BINARY_OP(XA, XB, OUT)                        ((void) 0)
#define DSC_TRACE_UNARY_OP(X, OUT)                              ((void) 0)
#define DSC_TRACE_UNARY_NO_OUT_OP(X)                            ((void) 0)
#define DSC_TRACE_UNARY_AXIS_OP(X, OUT, axis_, keep_dims_)      ((void) 0)
#define DSC_TRACE_FFT_OP(X, OUT, n_, axis_, type_, fwd_)        ((void) 0)
#define DSC_TRACE_PLAN_FFT(n_, fft_n_, fft_type_, dtype_)       ((void) 0)
#define DSC_TRACE_GET_IDX(X, indexes_, n_indexes_)              ((void) 0)
#define DSC_TRACE_GET_SLICE(X, slices_, n_slices_)              ((void) 0)
#define DSC_TRACE_SET_IDX(XA, XB, indexes_, n_indexes_)         ((void) 0)
#define DSC_TRACE_SET_SLICE(XA, XB, slices_, n_slices_)         ((void) 0)
#define DSC_TRACE_CAST_OP(X, new_dtype_)                        ((void) 0)
#define DSC_TRACE_RANDN_OP(shape_, n_dim_, dtype_)              ((void) 0)
#define DSC_TRACE_ARANGE_OP(n_, dtype_)                         ((void) 0)
#define DSC_TRACE_RESHAPE_OP(X, new_ndim_, new_shape_)          ((void) 0)
#define DSC_TRACE_CONCAT_OP(tensors_, axis_)                    ((void) 0)
#define DSC_TRACE_TRANSPOSE_OP(X, swap_axes_)                   ((void) 0)

#endif // DSC_ENABLE_TRACING

extern void dsc_internal_init_traces(u64 max_traces) noexcept;

extern void dsc_internal_free_traces() noexcept;

extern void dsc_internal_record_traces(bool record) noexcept;

extern void dsc_internal_dump_traces(const char *filename) noexcept;

extern void dsc_internal_clear_traces() noexcept;
