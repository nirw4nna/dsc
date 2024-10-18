// Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#include "dsc_tracing.h"

#if defined(DSC_ENABLE_TRACING)
#   include <cstdio>
#   include <cinttypes>         // PRIxPTR
#   include "dsc_backend.h"     // DSC_BACKED_NAMES
#   include "dsc_allocator.h"   // DSC_ALLOCATOR_NAMES

#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wunused-function"
// dsc_fft.h contains inline functions, since those are not used in this translation unit
// (as it should be since we only need the definition of dsc_fft_type) disable the warning.
#   include "dsc_fft.h"         // dsc_fft_type
#   pragma GCC diagnostic pop


dsc_trace_ctx *g_trace_ctx = nullptr;

void dsc_internal_init_traces(const u64 max_traces) noexcept {
    if (g_trace_ctx == nullptr) {
        DSC_LOG_DEBUG("max_traces=%ld", max_traces);
        g_trace_ctx = (dsc_trace_ctx *) malloc(sizeof(dsc_trace_ctx));
        g_trace_ctx->traces = (dsc_trace *) malloc(max_traces * sizeof(dsc_trace));
        g_trace_ctx->n_traces = 0;
        g_trace_ctx->max_traces = max_traces;
        g_trace_ctx->record = false;
    }
}

void dsc_internal_free_traces() noexcept {
    if (g_trace_ctx != nullptr) {
        free(g_trace_ctx->traces);
        free(g_trace_ctx);
        g_trace_ctx = nullptr;
    }
}

void dsc_internal_record_traces(const bool record) noexcept {
    g_trace_ctx->record = record;
}

static DSC_INLINE void dump_indexes(FILE *f, const int *indexes,
                                    const int n_indexes) noexcept {
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

static DSC_INLINE void dump_slices(FILE *f, const dsc_slice *slices,
                                   const int n_slices) noexcept {
    if (n_slices > 1) {
        fprintf(f, "\"[");
        for (int i = 0; i < n_slices; ++i) {
            fprintf(f, "%d:%d:%d",
                    slices[i].start,
                    slices[i].stop,
                    slices[i].step
            );
            if (i < n_slices - 1) fprintf(f, ", ");
        }
        fprintf(f, "]\"");
    } else {
        fprintf(f, "\"%d:%d:%d\"",
                slices[0].start,
                slices[0].stop,
                slices[0].step
        );
    }
}

static DSC_INLINE void dump_tensor_args(FILE *f, const dsc_tensor_args *t) noexcept {
    fprintf(f, R"({"shape": )");
    dump_indexes(f, t->shape, t->n_dim);
    fprintf(f, R"(, "dtype": "%s", "backend": "%s")",
            DSC_DTYPE_NAMES[t->dtype],
            DSC_BACKED_NAMES[t->backend]
    );
    if (t->addr != 0) fprintf(f, ", \"addr\": \"0x%" PRIxPTR "\"", t->addr);
    fprintf(f, "}");
}

static DSC_INLINE void dump_trace_args(FILE *f, const dsc_trace *t) noexcept {
    switch (t->type) {
        case DSC_OBJ_ALLOC:
        case DSC_OBJ_FREE: {
            const dsc_obj_alloc_args args = t->obj_alloc;
            fprintf(f, R"(, "args": {)");
            if (args.mem_size != 0) fprintf(f, R"("mem_size": %ld, )", args.mem_size);
            if (args.addr != 0) fprintf(f, "\"addr\": \"0x%" PRIxPTR "\", ", args.addr);

            fprintf(f, R"("allocator": "%s"})", DSC_ALLOCATOR_NAMES[args.allocator]);
            break;
        }
        case DSC_TENSOR_ALLOC:
        case DSC_TENSOR_FREE: {
            fprintf(f, R"(, "args": )");
            dump_tensor_args(f, &t->tensor_alloc.x);
            break;
        }
        case DSC_BINARY_OP: {
            const dsc_binary_args *args = &t->binary;
            fprintf(f, R"(, "args": {"xa": )");
            dump_tensor_args(f, &args->xa);
            fprintf(f, R"(, "xb": )");
            dump_tensor_args(f, &args->xb);
            if (args->with_out) {
                fprintf(f, R"(, "out": )");
                dump_tensor_args(f, &args->out);
            }
            fprintf(f, "}");
            break;
        }
        case DSC_UNARY_OP: {
            const dsc_unary_args *args = &t->unary;
            fprintf(f, R"(, "args": {"x": )");
            dump_tensor_args(f, &args->x);
            if (args->with_out) {
                fprintf(f, R"(, "out": )");
                dump_tensor_args(f, &args->out);
            }
            fprintf(f, "}");
            break;
        }
        case DSC_UNARY_NO_OUT_OP: {
            const dsc_unary_no_out_args *args = &t->unary_no_out;
            fprintf(f, R"(, "args": {"x": )");
            dump_tensor_args(f, &args->x);
            fprintf(f, "}");
            break;
        }
        case DSC_UNARY_AXIS_OP: {
            const dsc_unary_axis_args *args = &t->unary_axis;
            fprintf(f, R"(, "args": {"axis": %d, "keepdims": "%s", "x": )",
                    args->axis, args->keep_dims ? "True" : "False");
            dump_tensor_args(f, &args->x);
            if (args->with_out) {
                fprintf(f, R"(, "out": )");
                dump_tensor_args(f, &args->out);
            }
            fprintf(f, "}");
            break;
        }
        case DSC_FFT_OP: {
            const dsc_fft_args *args = &t->fft;
            fprintf(f, R"(, "args": {"type": "%s", "order": %d, "axis": %d, "x": )",
                    args->type == dsc_fft_type::COMPLEX ? (args->forward ? "FFT" : "IFFT") :
                                                        (args->forward ? "RFFT" : "IRFFT"),
                    args->n, args->axis);
            dump_tensor_args(f, &args->x);
            if (args->with_out) {
                fprintf(f, R"(, "out": )");
                dump_tensor_args(f, &args->out);
            }
            fprintf(f, "}");
            break;
        }
        case DSC_PLAN_FFT: {
            const dsc_plan_fft_args *args = &t->plan_fft;
            fprintf(f, R"(, "args": {"type": "%s", "n": %d, "order": %d, "dtype": "%s"})",
                    args->type == dsc_fft_type::COMPLEX ? "FFT" : "RFFT",
                    args->requested_n, args->fft_n,
                    DSC_DTYPE_NAMES[args->dtype]);
            break;
        }
        case DSC_GET_IDX: {
            const dsc_get_idx_args *args = &t->get_idx;
            fprintf(f, R"(, "args": {"x": )");
            dump_tensor_args(f, &args->x);
            fprintf(f, R"(, "idx": )");
            dump_indexes(f, args->indexes, args->n_indexes);
            fprintf(f, "}");
            break;
        }
        case DSC_GET_SLICE: {
            const dsc_get_slice_args *args = &t->get_slice;
            fprintf(f, R"(, "args": {"x": )");
            dump_tensor_args(f, &args->x);
            fprintf(f, R"(, "idx": )");
            dump_slices(f, args->slices, args->n_slices);
            fprintf(f, "}");
            break;
        }
        case DSC_SET_IDX: {
            const dsc_set_idx_args *args = &t->set_idx;
            fprintf(f, R"(, "args": {"xa": )");
            dump_tensor_args(f, &args->xa);
            fprintf(f, R"(, "xb": )");
            dump_tensor_args(f, &args->xb);
            fprintf(f, R"(, "idx": )");
            dump_slices(f, args->indexes, args->n_indexes);
            fprintf(f, "}");
            break;
        }
        case DSC_SET_SLICE: {
            const dsc_set_slice_args *args = &t->set_slice;
            fprintf(f, R"(, "args": {"xa": )");
            dump_tensor_args(f, &args->xa);
            fprintf(f, R"(, "xb": )");
            dump_tensor_args(f, &args->xb);
            fprintf(f, R"(, "idx": )");
            dump_slices(f, args->slices, args->n_slices);
            fprintf(f, "}");
            break;
        }
        case DSC_CAST_OP: {
            const dsc_cast_args *args = &t->cast;
            fprintf(f, R"(, "args": {"x": )");
            dump_tensor_args(f, &args->x);
            fprintf(f, R"(, "new_dtype": "%s"})",
                    DSC_DTYPE_NAMES[args->new_dtype]);
            break;
        }
        case DSC_RANDN_OP: {
            const dsc_randn_args *args = &t->randn;
            fprintf(f, R"(, "args": {"shape": )");
            dump_indexes(f, args->shape, args->n_dim);
            fprintf(f, R"(, "dtype": "%s"})",
                    DSC_DTYPE_NAMES[args->dtype]);
            break;
        }
        case DSC_ARANGE_OP: {
            const dsc_arange_args *args = &t->arange;
            fprintf(f, R"(, "args": {"n": %d, "dtype": "%s"})",
                    args->n, DSC_DTYPE_NAMES[args->dtype]);
            break;
        }
            DSC_INVALID_CASE("unknown trace type %d", t->type);
    }
}

void dsc_internal_dump_traces(const char *filename) noexcept {
    FILE *f = fopen(filename, "wt");
    DSC_ASSERT(f != nullptr);

    fprintf(f, "[\n");
    for (u64 i = 0; i < g_trace_ctx->n_traces; ++i) {
        dsc_trace *trace = &g_trace_ctx->traces[i];
        fprintf(f, "\t" R"({"name": "%s", "cat": "%s", "ph": "%c", "ts": %ld, "pid": %d, "tid": %ld)",
                trace->name, trace->cat, trace->phase, trace->ts, trace->pid, trace->tid);

        dump_trace_args(f, trace);

        fprintf(f, "}");
        if (i < g_trace_ctx->n_traces - 1) fprintf(f, ",");
        fprintf(f, "\n");
    }
    fprintf(f, "]");
    fclose(f);

    DSC_LOG_INFO("exported Perfetto-compatible traces to \"%s\"", filename);
}

void dsc_internal_clear_traces() noexcept {
    g_trace_ctx->n_traces = 0;
}

#else

void dsc_internal_init_traces(const u64 max_traces) noexcept {
    DSC_UNUSED(max_traces);
}

void dsc_internal_free_traces() noexcept {}

void dsc_internal_record_traces(const bool record) noexcept {
    DSC_UNUSED(record);
}

void dsc_internal_dump_traces(const char *filename) noexcept {
    DSC_UNUSED(filename);
}

void dsc_internal_clear_traces() noexcept {}

#endif // DSC_ENABLE_TRACING