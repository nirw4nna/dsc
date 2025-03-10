// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "dsc.h"

#if defined(DSC_ENABLE_TRACING)

#include <cstring>
#include <unistd.h>     // getpid()
#include <pthread.h>    // pthread_self()
#include <cinttypes>    // PRIxPTR


#define DSC_TRACE_NAME_MAX  ((int) 32)
#define DSC_TRACE_CAT_MAX   ((int) 16)

#define DSC_INSERT_TYPED_TRACE(T, cat_, type_) \
    dsc_trace_tracker<T> trace__{ctx->trace_ctx, __FUNCTION__, (cat_), (type_), &args__}

#define DSC_TRACE_SET_TENSOR(X, field)                                                         \
    args__.field.n_dim = (X)->n_dim;                                                           \
    memcpy(args__.field.shape, &dsc_tensor_get_dim((X), 0), (X)->n_dim * sizeof(*(X)->shape)); \
    args__.field.dtype = (X)->dtype;                                                           \
    args__.field.device = (X)->device;                                                         \
    args__.field.addr = (uintptr_t) (X)

#define DSC_TRACE_TENSOR_NEW(shape_, n_dim_, dtype_, device_)        \
    dsc_tensor_alloc_args args__{};                                  \
    memcpy(&args__.x.shape, (shape_), (n_dim_) * sizeof(*(shape_))); \
    args__.x.n_dim = (n_dim_);                                       \
    args__.x.dtype = (dtype_);                                       \
    args__.x.device = (device_);                                     \
    args__.x.addr = 0;                                               \
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

#define DSC_TRACE_MASK_OP(X, MASK, value_) \
    dsc_mask_args args__{};                \
    DSC_TRACE_SET_TENSOR(X, x);            \
    DSC_TRACE_SET_TENSOR(MASK, mask);      \
    args__.value = (value_);               \
    DSC_INSERT_TYPED_TRACE(dsc_mask_args, "op;mask", DSC_MASK_OP)

#define DSC_TRACE_UNARY_OP(X, OUT)      \
    dsc_unary_args args__{};            \
    DSC_TRACE_SET_TENSOR(X, x);         \
    if ((OUT) != nullptr) {             \
        DSC_TRACE_SET_TENSOR(OUT, out); \
    }                                   \
    args__.with_out = (OUT) != nullptr; \
    DSC_INSERT_TYPED_TRACE(dsc_unary_args, "op;unary", DSC_UNARY_OP)

#define DSC_TRACE_UNARY_AXIS_OP(X, OUT, axis_, keep_dims_) \
    dsc_unary_axis_args args__{};                          \
    DSC_TRACE_SET_TENSOR(X, x);                            \
    if ((OUT) != nullptr) {                                \
        DSC_TRACE_SET_TENSOR(OUT, out);                    \
    }                                                      \
    args__.with_out = (OUT) != nullptr;                    \
    args__.axis = (axis_);                                 \
    args__.keep_dims = (keep_dims_);                       \
    DSC_INSERT_TYPED_TRACE(dsc_unary_axis_args, "op;reduce", DSC_UNARY_AXIS_OP)

#define DSC_TRACE_GET_IDX(X, indexes_, n_indexes_)                          \
    dsc_get_idx_args args__{};                                              \
    DSC_TRACE_SET_TENSOR(X, x);                                             \
    memcpy(args__.indexes, (indexes_), (n_indexes_) * sizeof(*(indexes_))); \
    args__.n_indexes = (n_indexes_);                                        \
    DSC_INSERT_TYPED_TRACE(dsc_get_idx_args, "idx;get", DSC_GET_IDX)

#define DSC_TRACE_GET_SLICE(X, slices_, n_slices_)  \
    dsc_get_slice_args args__{};                    \
    DSC_TRACE_SET_TENSOR(X, x);                     \
    memcpy(args__.slices, (slices_), (n_slices_) * sizeof(*(slices_))); \
    args__.n_slices = (n_slices_);  \
    DSC_INSERT_TYPED_TRACE(dsc_get_slice_args, "slice;get", DSC_GET_SLICE)

#define DSC_TRACE_GET_TENSOR(X, INDEXES)    \
    dsc_get_tensor_args args__{};           \
    DSC_TRACE_SET_TENSOR(X, x);             \
    DSC_TRACE_SET_TENSOR(INDEXES, indexes); \
    DSC_INSERT_TYPED_TRACE(dsc_get_tensor_args, "tensor;get", DSC_GET_TENSOR)

#define DSC_TRACE_SET_IDX(XA, XB, indexes_, n_indexes_)                     \
    dsc_set_idx_args args__{};                                              \
    DSC_TRACE_SET_TENSOR(XA, xa);                                           \
    DSC_TRACE_SET_TENSOR(XB, xb);                                           \
    memcpy(args__.indexes, (indexes_), (n_indexes_) * sizeof(*(indexes_))); \
    args__.n_indexes = (n_indexes_);                                        \
    DSC_INSERT_TYPED_TRACE(dsc_set_idx_args, "idx;set", DSC_SET_IDX)

#define DSC_TRACE_SET_SLICE(XA, XB, slices_, n_slices_)                 \
    dsc_set_slice_args args__{};                                        \
    DSC_TRACE_SET_TENSOR(XA, xa);                                       \
    DSC_TRACE_SET_TENSOR(XB, xb);                                       \
    memcpy(args__.slices, (slices_), (n_slices_) * sizeof(*(slices_))); \
    args__.n_slices = (n_slices_);                                      \
    DSC_INSERT_TYPED_TRACE(dsc_set_slice_args, "slice;set", DSC_SET_SLICE)

#define DSC_TRACE_CAST_OP(X, new_dtype_)    \
    dsc_cast_args args__{};                 \
    DSC_TRACE_SET_TENSOR(X, x);             \
    args__.new_dtype = (new_dtype_);        \
    DSC_INSERT_TYPED_TRACE(dsc_cast_args, "op;cast", DSC_CAST_OP)

#define DSC_TRACE_RANDN_OP(shape_, n_dim_, dtype_)                 \
    dsc_randn_args args__{};                                       \
    memcpy(&args__.shape, (shape_), (n_dim_) * sizeof(*(shape_))); \
    args__.n_dim = (n_dim_);                                       \
    args__.dtype = (dtype_);                                       \
    DSC_INSERT_TYPED_TRACE(dsc_randn_args, "op;randn", DSC_RANDN_OP)

#define DSC_TRACE_TOPK_OP(X, k_, axis_, largest_) \
    dsc_topk_args args__{};                       \
    DSC_TRACE_SET_TENSOR(X, x);                   \
    args__.k = (k_);                              \
    args__.axis = (axis_);                        \
    args__.largest = (largest_);                  \
    DSC_INSERT_TYPED_TRACE(dsc_topk_args, "op;topk", DSC_TOPK_OP)

#define DSC_TRACE_MULTINOMIAL_OP(X, num_samples_) \
    dsc_multinomial_args args__{};                \
    DSC_TRACE_SET_TENSOR(X, x);                   \
    args__.num_samples = (num_samples_);          \
    DSC_INSERT_TYPED_TRACE(dsc_multinomial_args, "op;multinomial", DSC_MULTINOMIAL_OP)

#define DSC_TRACE_ARANGE_OP(n_, dtype_) \
    dsc_arange_args args__{};           \
    args__.n = (n_);                    \
    args__.dtype = (dtype_);            \
    DSC_INSERT_TYPED_TRACE(dsc_arange_args, "op;arange", DSC_ARANGE_OP)

#define DSC_TRACE_COPY_OP(X, data_, nb_, data_device_) \
    dsc_copy_args args__{};                            \
    DSC_TRACE_SET_TENSOR(X, x);                        \
    args__.data = (uintptr_t) (data_);                 \
    args__.nb = (nb_);                                 \
    args__.data_device = (data_device_);               \
    DSC_INSERT_TYPED_TRACE(dsc_copy_args, "op;copy", DSC_COPY_OP)

#define DSC_TRACE_CONCAT_OP(tensors_, axis_) \
    dsc_concat_args args__{};                \
    args__.tensors = (tensors_);             \
    args__.axis_ = (axis_);                  \
    DSC_INSERT_TYPED_TRACE(dsc_concat_args, "op;concat", DSC_CONCAT_OP)

#define DSC_TRACE_TRANSPOSE_OP(X, swap_axes_)                                   \
    dsc_transpose_args args__{};                                                \
    DSC_TRACE_SET_TENSOR(X, x);                                                 \
    memcpy(args__.swap_axes, (swap_axes_), (X)->n_dim * sizeof(*(swap_axes_))); \
    DSC_INSERT_TYPED_TRACE(dsc_transpose_args, "op;transpose", DSC_TRANSPOSE_OP)


#define TYPED_FILL(NAME, ARGS)                       \
    if constexpr (dsc_is_type<T, ARGS>()) {          \
        const ARGS *args_ = (const ARGS *) args;     \
        memcpy(&trace->NAME, args_, sizeof(*args_)); \
    }

#define TYPED_DUMP(TYPE, ARGS) \
    case TYPE:                 \
        trace->ARGS.dump(f);   \
        break

#define DUMP() DSC_INLINE void dump(FILE *f) const

namespace internal::tracing {
DSC_INLINE void dump_indexes(FILE *f, const int *indexes,
                             const int n_indexes) {
    if (n_indexes > 1) {
        fprintf(f, "\"[");
        for (int i = 0; i < n_indexes; ++i) {
            fprintf(f, "%d", indexes[i]);
            if (i < n_indexes - 1) fprintf(f, ", ");
        }
        fprintf(f, "]\"");
    } else {
        fprintf(f, "%d", indexes[0]);
    }
}

DSC_INLINE void dump_slices(FILE *f, const dsc_slice *slices,
                            const int n_slices) {
    if (n_slices > 1) {
        fprintf(f, "\"[");
        for (int i = 0; i < n_slices; ++i) {
            fprintf(f, "%d:%d:%d",
                    slices[i].start,
                    slices[i].stop,
                    slices[i].step);
            if (i < n_slices - 1) fprintf(f, ", ");
        }
        fprintf(f, "]\"");
    } else {
        fprintf(f, "\"%d:%d:%d\"",
                slices[0].start,
                slices[0].stop,
                slices[0].step);
    }
}
}

enum dsc_trace_phase : char {
    BEGIN = 'B',
    END = 'E',
    COMPLETE = 'X'
};

enum dsc_trace_type : u8 {
    DSC_TRACE_EMPY, // Trace without any args
    DSC_TENSOR_ALLOC,
    DSC_TENSOR_FREE,
    DSC_UNARY_OP,
    DSC_UNARY_AXIS_OP,
    DSC_BINARY_OP,
    DSC_MASK_OP,
    DSC_GET_IDX,
    DSC_GET_SLICE,
    DSC_GET_TENSOR,
    DSC_SET_IDX,
    DSC_SET_SLICE,
    DSC_CAST_OP,
    DSC_RANDN_OP,
    DSC_TOPK_OP,
    DSC_MULTINOMIAL_OP,
    DSC_ARANGE_OP,
    DSC_COPY_OP,
    DSC_CONCAT_OP,
    DSC_TRANSPOSE_OP,
};

struct dsc_empty_args {};

struct dsc_tensor_args {
    int shape[DSC_MAX_DIMS];
    uintptr_t addr;
    int n_dim;
    dsc_device_type device;
    dsc_dtype dtype;

    DUMP() {
        fprintf(f, R"({"shape": )");
        internal::tracing::dump_indexes(f, shape, n_dim);
        fprintf(f, R"(, "dtype": "%s", "device": "%s")",
            DSC_DTYPE_NAMES[dtype],
            DSC_DEVICE_NAMES[device]
        );
        if (addr != 0) fprintf(f, ", \"addr\": \"0x%" PRIxPTR "\"", addr);
        fprintf(f, "}");
    }
};

struct dsc_tensor_alloc_args {
    dsc_tensor_args x;

    DUMP() {
        fprintf(f, R"("x": )");
        x.dump(f);
    }
};

struct dsc_binary_args {
    dsc_tensor_args xa, xb, out;
    bool with_out;

    DUMP() {
        fprintf(f, R"("xa": )");
        xa.dump(f);
        fprintf(f, R"(, "xb": )");
        xb.dump(f);
        if (with_out) {
            fprintf(f, R"(, "out": )");
            out.dump(f);
        }
    }
};

struct dsc_mask_args {
    dsc_tensor_args x, mask;
    f64 value;

    DUMP() {
        fprintf(f, R"("x": )");
        x.dump(f);
        fprintf(f, R"(, "mask": )");
        mask.dump(f);
        fprintf(f, R"(, "value": %f)", value);
    }
};

struct dsc_unary_args {
    dsc_tensor_args x, out;
    bool with_out;

    DUMP() {
        fprintf(f, R"("x": )");
        x.dump(f);
        if (with_out) {
            fprintf(f, R"(, "out": )");
            out.dump(f);
        }
    }
};

struct dsc_unary_axis_args {
    dsc_tensor_args x, out;
    int axis;
    bool keep_dims, with_out;

    DUMP() {
        fprintf(f, R"("axis": %d, "keepdims": "%s", "x": )", axis, keep_dims ? "True" : "False");
        x.dump(f);
        if (with_out) {
            fprintf(f, R"(, "out": )");
            out.dump(f);
        }
    }
};

struct dsc_get_idx_args {
    dsc_tensor_args x;
    int indexes[DSC_MAX_DIMS];
    int n_indexes;

    DUMP() {
        fprintf(f, R"("x": )");
        x.dump(f);
        fprintf(f, R"(, "idx": )");
        internal::tracing::dump_indexes(f, indexes, n_indexes);
    }
};

struct dsc_get_slice_args {
    dsc_tensor_args x;
    dsc_slice slices[DSC_MAX_DIMS];
    int n_slices;

    DUMP() {
        fprintf(f, R"("x": )");
        x.dump(f);
        fprintf(f, R"(, "slice": )");
        internal::tracing::dump_slices(f, slices, n_slices);
    }
};

struct dsc_get_tensor_args {
    dsc_tensor_args x, indexes;

    DUMP() {
        fprintf(f, R"("x": )");
        x.dump(f);
        fprintf(f, R"(, "idx": )");
        indexes.dump(f);
    }
};

struct dsc_set_idx_args {
    dsc_tensor_args xa, xb;
    dsc_slice indexes[DSC_MAX_DIMS];
    int n_indexes;

    DUMP() {
        fprintf(f, R"("xa": )");
        xa.dump(f);
        fprintf(f, R"(, "xb": )");
        xb.dump(f);
        fprintf(f, R"(, "idx": )");
        internal::tracing::dump_slices(f, indexes, n_indexes);
    }
};

struct dsc_set_slice_args {
    dsc_tensor_args xa, xb;
    dsc_slice slices[DSC_MAX_DIMS];
    int n_slices;

    DUMP() {
        fprintf(f, R"("xa": )");
        xa.dump(f);
        fprintf(f, R"(, "xb": )");
        xb.dump(f);
        fprintf(f, R"(, "slice": )");
        internal::tracing::dump_slices(f, slices, n_slices);
    }
};

struct dsc_cast_args {
    dsc_tensor_args x;
    dsc_dtype new_dtype;

    DUMP() {
        fprintf(f, R"("x": )");
        x.dump(f);
        fprintf(f, R"(, "new_dtype": "%s")", DSC_DTYPE_NAMES[new_dtype]);
    }
};

struct dsc_randn_args {
    int shape[DSC_MAX_DIMS];
    int n_dim;
    dsc_dtype dtype;

    DUMP() {
        fprintf(f, R"("shape": )");
        internal::tracing::dump_indexes(f, shape, n_dim);
        fprintf(f, R"(, "dtype": "%s")", DSC_DTYPE_NAMES[dtype]);
    }
};

struct dsc_topk_args {
    dsc_tensor_args x;
    int k, axis;
    bool largest;

    DUMP() {
        fprintf(f, R"("x": )");
        x.dump(f);
        fprintf(f, R"(, "k": %d, "axis": %d, "largest": "%s")", k, axis, largest ? "True" : "False");
    }
};

struct dsc_multinomial_args {
    dsc_tensor_args x;
    int num_samples;

    DUMP() {
        fprintf(f, R"("x": )");
        x.dump(f);
        fprintf(f, R"(, "num_samples": %d)", num_samples);
    }
};

struct dsc_arange_args {
    int n;
    dsc_dtype dtype;

    DUMP() {
        fprintf(f, R"("n": %d, "dtype": "%s")", n, DSC_DTYPE_NAMES[dtype]);
    }
};

struct dsc_copy_args {
    dsc_tensor_args x;
    uintptr_t data;
    usize nb;
    dsc_device_type data_device;

    DUMP() {
        fprintf(f, R"("x": )");
        x.dump(f);
        fprintf(f, ", \"addr\": \"0x%" PRIxPTR "\"", data);
        fprintf(f, R"(, "nb": %ld, "data_device": "%s")", nb, DSC_DEVICE_NAMES[data_device]);
    }
};

struct dsc_concat_args {
    int tensors;
    int axis;

    DUMP() {
        char axis_str[16];
        if (axis == DSC_VALUE_NONE) snprintf(axis_str, 16, "Flatten");
        else snprintf(axis_str, 16, "%d", axis);
        fprintf(f, R"("tensors": %d, "axis": "%s")", tensors, axis_str);
    }
};

struct dsc_transpose_args {
    dsc_tensor_args x;
    int swap_axes[DSC_MAX_DIMS];

    DUMP() {
        fprintf(f, R"("x": )");
        x.dump(f);
        fprintf(f, R"(, "axes": )");
        internal::tracing::dump_indexes(f, swap_axes, x.n_dim);
    }
};


struct dsc_trace {
    char name[DSC_TRACE_NAME_MAX], cat[DSC_TRACE_CAT_MAX];
    u64 tid, ts; // Timestamp of the event in us
    int pid;
    dsc_trace_phase phase;
    dsc_trace_type type;
    union {
        dsc_empty_args empty;
        dsc_tensor_alloc_args tensor_alloc;
        dsc_unary_args unary;
        dsc_unary_axis_args unary_axis;
        dsc_binary_args binary;
        dsc_mask_args mask;
        dsc_get_idx_args get_idx;
        dsc_get_slice_args get_slice;
        dsc_get_tensor_args get_tensor;
        dsc_set_idx_args set_idx;
        dsc_set_slice_args set_slice;
        dsc_cast_args cast;
        dsc_randn_args randn;
        dsc_topk_args topk;
        dsc_multinomial_args multinomial;
        dsc_arange_args arange;
        dsc_copy_args copy;
        dsc_concat_args concat;
        dsc_transpose_args transpose;
    };
};

struct dsc_trace_ctx {
    dsc_trace *traces;
    u64 n_traces, max_traces;
    bool record;
};

namespace internal::tracing {
DSC_INLINE u64 time_us() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (u64) (ts.tv_sec * 1'000'000ULL) + (u64) (ts.tv_nsec / 1'000ULL);
}

template<typename T = dsc_empty_args>
DSC_INLINE void fill_trace(dsc_trace *trace, const char *name,
                           const char *cat, const dsc_trace_phase phase,
                           const dsc_trace_type type,
                           const T *args = nullptr) {
    strncpy(trace->name, name, DSC_TRACE_NAME_MAX);
    strncpy(trace->cat, cat, DSC_TRACE_CAT_MAX);
    trace->pid = getpid();
    trace->tid = pthread_self();
    trace->phase = phase;
    trace->type = type;

    TYPED_FILL(tensor_alloc, dsc_tensor_alloc_args)
    TYPED_FILL(binary, dsc_binary_args)
    TYPED_FILL(mask, dsc_mask_args)
    TYPED_FILL(unary, dsc_unary_args)
    TYPED_FILL(unary_axis, dsc_unary_axis_args)
    TYPED_FILL(cast, dsc_cast_args)
    TYPED_FILL(transpose, dsc_transpose_args)
    TYPED_FILL(copy, dsc_copy_args)
    TYPED_FILL(concat, dsc_concat_args)
    TYPED_FILL(get_idx, dsc_get_idx_args)
    TYPED_FILL(get_slice, dsc_get_slice_args)
    TYPED_FILL(get_tensor, dsc_get_tensor_args)
    TYPED_FILL(set_idx, dsc_set_idx_args)
    TYPED_FILL(set_slice, dsc_set_slice_args)
    TYPED_FILL(randn, dsc_randn_args)
    TYPED_FILL(topk, dsc_topk_args)
    TYPED_FILL(multinomial, dsc_multinomial_args)
    TYPED_FILL(arange, dsc_arange_args)
}

DSC_INLINE bool can_trace(dsc_trace_ctx *ctx) {
    return ctx->record &&
           ctx->n_traces < ctx->max_traces;
}

DSC_INLINE dsc_trace *next_available_trace(dsc_trace_ctx *ctx) {
    return &ctx->traces[ctx->n_traces++];
}
}


template<typename T>
struct dsc_trace_tracker {
    dsc_trace_tracker(dsc_trace_ctx *ctx,
                      const char *name,
                      const char *cat,
                      const dsc_trace_type type,
                      const T *args) : ctx_(ctx), args_(args), name_(name),
                                       cat_(cat), type_(type) {
        using namespace internal::tracing;

        if (can_trace(ctx_)) {
            dsc_trace *t = next_available_trace(ctx_);
            fill_trace(t, name_, cat_, BEGIN, type_, args);
            t->ts = time_us();
        }
    }

    ~dsc_trace_tracker() {
        using namespace internal::tracing;

        if (can_trace(ctx_)) {
            dsc_trace *t = next_available_trace(ctx_);
            t->ts = time_us();
            fill_trace(t, name_, cat_, END, type_, args_);
        }
    }

private:
    dsc_trace_ctx *ctx_;
    const T *args_;
    const char *name_, *cat_;
    dsc_trace_type type_;
};


static DSC_INLINE dsc_trace_ctx *dsc_tracing_init() {
    static dsc_trace_ctx ctx = {
        .traces = (dsc_trace *) malloc(DSC_MAX_TRACES * sizeof(dsc_trace)),
        .n_traces = 0,
        .max_traces = DSC_MAX_TRACES,
        .record = false,
    };
    return &ctx;
}

static DSC_INLINE void dsc_tracing_free(dsc_trace_ctx *ctx) {
    free(ctx->traces);
}

static DSC_INLINE void dsc_tracing_record(dsc_trace_ctx *ctx, const bool record) {
    ctx->record = record;
}

static DSC_INLINE void dsc_tracing_clear(dsc_trace_ctx *ctx) {
    ctx->n_traces = 0;
}

static DSC_INLINE void dsc_tracing_insert(dsc_trace_ctx *ctx,
                                          const char *name,
                                          const char *cat,
                                          const u64 ts,
                                          const dsc_trace_phase phase) {
    using namespace internal::tracing;

    if (can_trace(ctx)) {
        dsc_trace *trace = next_available_trace(ctx);
        fill_trace(trace, name, cat, phase, DSC_TRACE_EMPY);
        trace->ts = ts;
    }
}

static dsc_traces dsc_tracing_get(dsc_trace_ctx *ctx) {
    return {.traces = ctx->traces, .n_traces = ctx->n_traces};
}

static DSC_INLINE void dsc_tracing_dump(dsc_trace_ctx *ctx, const char *filename) {
    FILE *f = fopen(filename, "wt");
    DSC_ASSERT(f != nullptr);

    fprintf(f, "[\n");
    for (u64 i = 0; i < ctx->n_traces; ++i) {
        dsc_trace *trace = &ctx->traces[i];
        fprintf(f, "\t" R"({"name": "%s", "cat": "%s", "ph": "%c", "ts": %ld, "pid": %d, "tid": %ld)",
                trace->name, trace->cat, trace->phase, trace->ts, trace->pid, trace->tid);

        if (trace->type != DSC_TRACE_EMPY) {
            fprintf(f, R"(, "args": {)");
            switch (trace->type) {
                TYPED_DUMP(DSC_TENSOR_ALLOC, tensor_alloc);
                TYPED_DUMP(DSC_TENSOR_FREE, tensor_alloc);
                TYPED_DUMP(DSC_UNARY_OP, unary);
                TYPED_DUMP(DSC_UNARY_AXIS_OP, unary_axis);
                TYPED_DUMP(DSC_BINARY_OP, binary);
                TYPED_DUMP(DSC_MASK_OP, mask);
                TYPED_DUMP(DSC_GET_IDX, get_idx);
                TYPED_DUMP(DSC_GET_SLICE, get_slice);
                TYPED_DUMP(DSC_GET_TENSOR, get_tensor);
                TYPED_DUMP(DSC_SET_IDX, set_idx);
                TYPED_DUMP(DSC_SET_SLICE, set_slice);
                TYPED_DUMP(DSC_CAST_OP, cast);
                TYPED_DUMP(DSC_RANDN_OP, randn);
                TYPED_DUMP(DSC_TOPK_OP, topk);
                TYPED_DUMP(DSC_MULTINOMIAL_OP, multinomial);
                TYPED_DUMP(DSC_ARANGE_OP, arange);
                TYPED_DUMP(DSC_COPY_OP, copy);
                TYPED_DUMP(DSC_CONCAT_OP, concat);
                TYPED_DUMP(DSC_TRANSPOSE_OP, transpose);

                DSC_INVALID_CASE("unknown trace type=%d", trace->type);
            }
            fprintf(f, "}");
        }

        fprintf(f, "}");
        if (i < ctx->n_traces - 1) fprintf(f, ",");
        fprintf(f, "\n");
    }
    fprintf(f, "]");
    fclose(f);

    DSC_LOG_INFO("exported Perfetto-compatible traces to \"%s\"", filename);
}

#undef TYPED_FILL
#undef TYPED_DUMP
#undef DUMP

#else

using dsc_trace_ctx = nullptr_t;

#define DSC_TRACE_TENSOR_NEW(shape_, n_dim_, dtype_, backend_)  ((void) 0)
#define DSC_TRACE_TENSOR_FREE(X)                                ((void) 0)
#define DSC_TRACE_BINARY_OP(XA, XB, OUT)                        ((void) 0)
#define DSC_TRACE_MASK_OP(X, MASK, value_)                      ((void) 0)
#define DSC_TRACE_UNARY_OP(X, OUT)                              ((void) 0)
#define DSC_TRACE_UNARY_AXIS_OP(X, OUT, axis_, keep_dims_)      ((void) 0)
#define DSC_TRACE_GET_IDX(X, indexes_, n_indexes_)              ((void) 0)
#define DSC_TRACE_GET_SLICE(X, slices_, n_slices_)              ((void) 0)
#define DSC_TRACE_GET_TENSOR(X, INDEXES)                        ((void) 0)
#define DSC_TRACE_SET_IDX(XA, XB, indexes_, n_indexes_)         ((void) 0)
#define DSC_TRACE_SET_SLICE(XA, XB, slices_, n_slices_)         ((void) 0)
#define DSC_TRACE_CAST_OP(X, new_dtype_)                        ((void) 0)
#define DSC_TRACE_RANDN_OP(shape_, n_dim_, dtype_)              ((void) 0)
#define DSC_TRACE_TOPK_OP(X, k_, axis_, largest_)               ((void) 0)
#define DSC_TRACE_MULTINOMIAL_OP(X, num_samples_)               ((void) 0)
#define DSC_TRACE_ARANGE_OP(n_, dtype_)                         ((void) 0)
#define DSC_TRACE_COPY_OP(X, data_, nb_, data_device_)          ((void) 0)
#define DSC_TRACE_CONCAT_OP(tensors_, axis_)                    ((void) 0)
#define DSC_TRACE_TRANSPOSE_OP(X, swap_axes_)                   ((void) 0)


static DSC_INLINE dsc_trace_ctx *dsc_tracing_init() {
    return nullptr;
}

static DSC_INLINE void dsc_tracing_free(dsc_trace_ctx *ctx) {
    DSC_UNUSED(ctx);
}

static DSC_INLINE void dsc_tracing_record(dsc_trace_ctx *ctx, const bool record) {
    DSC_UNUSED(ctx);
    DSC_UNUSED(record);
}

static DSC_INLINE void dsc_tracing_clear(dsc_trace_ctx *ctx) {
    DSC_UNUSED(ctx);
}

static DSC_INLINE void dsc_tracing_insert(dsc_trace_ctx *ctx,
                                          const char *name,
                                          const char *cat,
                                          const u64 ts,
                                          const dsc_trace_phase phase) {
    DSC_UNUSED(ctx);
    DSC_UNUSED(name);
    DSC_UNUSED(cat);
    DSC_UNUSED(ts);
    DSC_UNUSED(phase);
}

static dsc_traces dsc_tracing_get(dsc_trace_ctx *ctx) {
    DSC_UNUSED(ctx);
    return {};
}

static void dsc_tracing_dump(dsc_trace_ctx *ctx, const char *filename) {
    DSC_UNUSED(ctx);
    DSC_UNUSED(filename);
}

#endif