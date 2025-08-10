// Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "dsc.h"
#include <cstdlib>
#include <ctime>        // timespec
#include <cinttypes>    // PRIxPTR
#include <cstring>


#define DSC_TRACE_SET_TENSOR(X, field)                                                         \
    args__.field.n_dim = (X)->n_dim;                                                           \
    args__.field.ne = (X)->ne;                                                                 \
    memcpy(args__.field.shape, &dsc_tensor_get_dim((X), 0), (X)->n_dim * sizeof(*(X)->shape)); \
    args__.field.dtype = (X)->dtype;                                                           \
    args__.field.device = (X)->device;                                                         \
    args__.field.addr = (uintptr_t) (X)

#define TYPED_FILL(NAME, ARGS)                       \
    if constexpr (dsc_is_type<T, ARGS>()) {          \
        const ARGS *args_ = (const ARGS *) args;     \
        memcpy(&trace->NAME, args_, sizeof(*args_)); \
    }

#define TYPED_DUMP(TYPE, ARGS) \
    case TYPE:                 \
        trace->ARGS.json_dump(f);   \
        break


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

struct dsc_empty_args {
    static DSC_INLINE u64 rw_bytes() { return 0; }
    static DSC_INLINE void json_dump(FILE *) {}
};

struct dsc_tensor_args {
    int shape[DSC_MAX_DIMS];
    uintptr_t addr;
    int n_dim, ne;
    dsc_device_type device;
    dsc_dtype dtype;

    DSC_INLINE u64 rw_bytes() const { return ne * DSC_DTYPE_SIZE[dtype]; }
    DSC_INLINE void json_dump(FILE *f) const {
        fprintf(f, R"({"shape":)");
        internal::tracing::dump_indexes(f, shape, n_dim);
        fprintf(f, R"(,"dtype":"%s","device":"%s")",
                DSC_DTYPE_NAMES[dtype],
                DSC_DEVICE_NAMES[device]);
        if (addr != 0) fprintf(f, ",\"addr\":\"0x%" PRIxPTR "\"", addr);
        fprintf(f, "}");
    }
};

struct dsc_cast_args {
    dsc_tensor_args x;
    dsc_dtype new_dtype;

    DSC_INLINE u64 rw_bytes() const { return x.ne * DSC_DTYPE_SIZE[x.dtype] + x.ne * DSC_DTYPE_SIZE[new_dtype]; }
    DSC_INLINE void json_dump(FILE *f) const {
        fprintf(f, R"(,"x":)");
        x.json_dump(f);
        fprintf(f, R"(,"new_dtype":"%s")", DSC_DTYPE_NAMES[new_dtype]);
    }
};

enum dsc_trace_type : u8 {
    DSC_TRACE_EMPY, // Trace without any args
    DSC_TRACE_CUSTOM,
    DSC_TENSOR_ALLOC,
    DSC_TENSOR_FREE,
    // DSC_UNARY_OP,
    // DSC_UNARY_AXIS_OP,
    // DSC_BINARY_OP,
    // DSC_MATMUL_OP,
    // DSC_MASK_OP,
    // DSC_OUTER_OP,
    // DSC_WHERE_OP,
    // DSC_GET_IDX,
    // DSC_GET_SLICE,
    // DSC_GET_TENSOR,
    // DSC_SET_IDX,
    // DSC_SET_SLICE,
    DSC_CAST_OP,
    // DSC_RANDN_OP,
    // DSC_TOPK_OP,
    // DSC_MULTINOMIAL_OP,
    // DSC_ARANGE_OP,
    // DSC_REPEAT_OP,
    // DSC_COPY_OP,
    // DSC_CONCAT_OP,
    // DSC_TRANSPOSE_OP,
};

static constexpr const char *DSC_TRACE_CATEGORY[] = {
    "",
    "custom",
    "alloc",
    "free",
    "op;cast"
};


struct dsc_trace_common {
    char name[DSC_TRACE_NAME_MAX];
    u64 rw_bytes;
    u64 ingestion_time_us;

    dsc_trace_type type;
    union {
        //     dsc_empty_args empty;
        // dsc_tensor_alloc_args tensor_alloc;
        //     dsc_unary_args unary;
        //     dsc_unary_axis_args unary_axis;
        //     dsc_binary_args binary;
        //     dsc_matmul_args matmul;
        //     dsc_mask_args mask;
        //     dsc_outer_args outer;
        //     dsc_where_args where;
        //     dsc_get_idx_args get_idx;
        //     dsc_get_slice_args get_slice;
        //     dsc_get_tensor_args get_tensor;
        //     dsc_set_idx_args set_idx;
        //     dsc_set_slice_args set_slice;
        dsc_cast_args cast;
        //     dsc_randn_args randn;
        //     dsc_topk_args topk;
        //     dsc_multinomial_args multinomial;
        //     dsc_arange_args arange;
        //     dsc_repeat_args repeat;
        //     dsc_copy_args copy;
        //     dsc_concat_args concat;
        //     dsc_transpose_args transpose;
    };
};

namespace internal::tracing {
DSC_INLINE u64 time_us() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (u64) (ts.tv_sec * 1'000'000ULL) + (u64) (ts.tv_nsec / 1'000ULL);
}

template<typename T>
DSC_INLINE void advance_current_trace(dsc_trace_ctx *ctx) {
    ctx->current_trace_idx++;
    const int chunk = ctx->current_trace_idx / DSC_MAX_TRACES_PER_CHUNK;
    const int idx_in_chunk = ctx->current_trace_idx % DSC_MAX_TRACES_PER_CHUNK;

    if (chunk >= ctx->n_chunks || idx_in_chunk >= ctx->n_traces) {
        // We are done
        ctx->current_trace = nullptr;
        ctx->current_trace_idx = 0;
    }

    T **traces = (T **) ctx->traces;
    ctx->current_trace = &traces[chunk][idx_in_chunk];
}

template<typename T>
DSC_INLINE void traces_allocate_chunk(dsc_trace_ctx *ctx) {
    ctx->traces[ctx->n_chunks] = (T *) malloc(DSC_MAX_TRACES_PER_CHUNK * sizeof(T));
    ctx->n_chunks++;
    // Reset n_traces
    ctx->n_traces = 0;
}

template<typename T>
DSC_INLINE dsc_trace_ctx *init() {
    static dsc_trace_ctx ctx{
        .traces = {},
        .current_trace = nullptr,
        .current_trace_idx = 0,
        .n_chunks = 0,
        .n_traces = 0,
    };

    traces_allocate_chunk<T>(&ctx);
    T **traces = (T **) ctx.traces;
    ctx.current_trace = &traces[0][0];
    return &ctx;
}

DSC_INLINE void dispose(const dsc_trace_ctx *ctx) {
    for (int i = 0; i < ctx->n_chunks; i++) {
        free(ctx->traces[i]);
    }
}

template<typename T>
DSC_INLINE void check_if_full(dsc_trace_ctx *ctx) {
    if (ctx->n_traces >= DSC_MAX_TRACES_PER_CHUNK) {

        if (ctx->n_chunks >= DSC_MAX_CHUNKS) {
            DSC_LOG_FATAL("can't allocate any more traces!");
        }

        // Allocate a brand-new chunk
        traces_allocate_chunk<T>(ctx);
    }
}

template<typename T>
DSC_INLINE T *next_empty_trace(dsc_trace_ctx *ctx) {
    T **traces = (T **) ctx->traces;
    return &traces[ctx->n_chunks - 1][ctx->n_traces++];
}

template<typename T = dsc_empty_args>
DSC_INLINE void fill_trace(dsc_trace_common *trace,
                           const char *name,
                           const dsc_trace_type type,
                           const T *args = nullptr) {
    trace->ingestion_time_us = time_us();
    strncpy(trace->name, name, DSC_TRACE_NAME_MAX);
    trace->type = type;
    trace->rw_bytes = args->rw_bytes();

    // TYPED_FILL(tensor_alloc, dsc_tensor_alloc_args)
    TYPED_FILL(cast, dsc_cast_args)
}

DSC_INLINE void dump_trace_base(FILE *f, const dsc_trace_common *trace) {
    switch (trace->type) {
        TYPED_DUMP(DSC_CAST_OP, cast);
        default:
            break;
    }
}
}


#undef TYPED_FILL
#undef TYPED_DUMP
