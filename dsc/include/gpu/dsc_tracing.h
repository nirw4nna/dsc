// Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "dsc.h"
#include "dsc_tracing_common.h"
#include "gpu/dsc_gpu.h"


#define INSERT_TYPED_GPU_TRACE(T, type_, grid_dim_, block_dim_) \
    dsc_gpu_trace_tracker<T> trace__ { dev->trace_ctx, __FUNCTION__, (type_), (grid_dim_), (block_dim_), &args__ }

// NOTE: it's very similar to its CPU counterpart, maybe I can do better...
#define DSC_TRACE_CAST_OP(X, OUT, grid_dim_, block_dim_) \
    dsc_cast_args args__{};                              \
    DSC_TRACE_SET_TENSOR(X, x);                          \
    args__.new_dtype = (OUT)->dtype;                     \
    INSERT_TYPED_GPU_TRACE(dsc_cast_args, DSC_CAST_OP, grid_dim_, block_dim_)

struct dsc_gpu_trace {
    dsc_trace_common base;

    gpu_event start_event, stop_event;
    dim3 grid_dim, block_dim;
    f32 elapsed_ms;

    DSC_INLINE bool to_eval() const {
        return this->elapsed_ms <= 0.f;
    }
};

template<typename T>
struct dsc_gpu_trace_tracker {
    dsc_gpu_trace_tracker(dsc_trace_ctx *ctx,
                      const char *name,
                      const dsc_trace_type type,
                      const dim3 grid_dim,
                      const dim3 block_dim,
                      const T *args) : ctx_(ctx) {
        using namespace internal::tracing;

        check_if_full<dsc_gpu_trace>(ctx);
        trace_ = next_empty_trace<dsc_gpu_trace>(ctx);
        fill_trace(&trace_->base, name, type, args);
        gpu_event_create(&trace_->start_event);
        gpu_event_create(&trace_->stop_event);
        trace_->grid_dim = grid_dim;
        trace_->block_dim = block_dim;
        trace_->elapsed_ms = 0.f;
        gpu_event_record(trace_->start_event);
    }

    ~dsc_gpu_trace_tracker() {
        gpu_event_record(trace_->stop_event);
    }

private:
    dsc_trace_ctx *ctx_;
    dsc_gpu_trace *trace_;
};

static DSC_INLINE dsc_trace_ctx *dsc_gpu_tracing_init() {
    return internal::tracing::init<dsc_gpu_trace>();
}

static DSC_INLINE void dsc_gpu_tracing_dispose(const dsc_trace_ctx *ctx) {
    internal::tracing::dispose(ctx);
}

static void dsc_gpu_tracing_dump(void *trace, FILE *json_file) {
    dsc_gpu_trace *gpu_trace = (dsc_gpu_trace *) trace;

    if (gpu_trace->to_eval()) {
        // Make sure this is called once for each trace
        DSC_GPU_CHECK(gpu_event_synchronize(gpu_trace->stop_event));
        DSC_GPU_CHECK(gpu_event_elapsed(&gpu_trace->elapsed_ms, gpu_trace->start_event,gpu_trace->stop_event));
    }

    const dsc_trace_common *base = &gpu_trace->base;

    const f64 elapsed_ms = gpu_trace->elapsed_ms;
    const u64 elapsed_us = (u64) (elapsed_ms * 1e3);
    const f64 bandwidth = (f64) base->rw_bytes / (elapsed_ms * 1e-3 * DSC_GB(1));

    // So that we can align this
    char formatted_kernel_name[256];
    snprintf(formatted_kernel_name, 256,
             "%s<(%d,%d,%d), (%d,%d,%d)>",
             base->name,
             gpu_trace->grid_dim.x,
             gpu_trace->grid_dim.y,
             gpu_trace->grid_dim.z,
             gpu_trace->block_dim.x,
             gpu_trace->block_dim.y,
             gpu_trace->block_dim.z);

    // Console dumping
    printf("*** [%ld] \033[38;5;208mGPU\033[0m\t%-40s %.2fms (%6ldus)\t|\t%10.2fGB/s (%ldB)\n",
           base->ingestion_time_us,
           formatted_kernel_name,
           elapsed_ms,
           elapsed_us,
           bandwidth,
           base->rw_bytes);

    // json dumping
    fprintf(json_file, R"({"name":"%s","cat":"%s","ph":"X","ts":%ld,"dur":%ld,"pid":0,"tid":0)",
            base->name,
            DSC_TRACE_CATEGORY[base->type],
            base->ingestion_time_us,
            elapsed_us);
    fprintf(json_file, R"(,"args":{"bandwidth":"%.2fGB/s")", bandwidth);
    internal::tracing::dump_trace_base(json_file, base);
    fprintf(json_file, R"==(,"launch_config":{"grid":"(%d,%d,%d)","block":"(%d,%d,%d)"}})==",
            gpu_trace->grid_dim.x,
            gpu_trace->grid_dim.y,
            gpu_trace->grid_dim.z,
            gpu_trace->block_dim.x,
            gpu_trace->block_dim.y,
            gpu_trace->block_dim.z);
    fprintf(json_file, R"(})" ",\n");

    if (gpu_trace->to_eval()) {
        DSC_GPU_CHECK(gpu_event_destroy(gpu_trace->start_event));
        DSC_GPU_CHECK(gpu_event_destroy(gpu_trace->stop_event));
    }
}

static void dsc_gpu_next_trace(dsc_trace_ctx *ctx) {
    internal::tracing::advance_current_trace<dsc_gpu_trace>(ctx);
}

static void dsc_gpu_dump_json_metadata(FILE *json_file, void *extra_info) {
    const dsc_gpu_dev_info *dev_info = (dsc_gpu_dev_info *) extra_info;
    fprintf(json_file, R"({"name":"process_name","ph":"M","pid":%d,"tid":0,"args":{"name":"%s:%s"},"process_sort_index":100})" ",\n",
            dev_info->dev_idx,
            DSC_GPU_PLATFORM_NAMES[dev_info->platform],
            dev_info->name);
    fprintf(json_file, R"({"name":"thread_name","ph":"M","pid":%d,"tid":0,"args":{"name":"Stream"},"thread_sort_index":101})" ",\n",
            dev_info->dev_idx);
}
