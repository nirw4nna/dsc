// Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "dsc.h"
#include "dsc_tracing_common.h"

#if defined(DSC_TRACING)

#include "gpu/dsc_gpu.h"

#undef DSC_INSERT_TYPED_TRACE
#undef DSC_INSERT_NAMED_TRACE

#define DSC_INSERT_TYPED_TRACE(DEV, T, type_, grid_dim_, block_dim_) \
    dsc_gpu_trace_tracker<T> trace__ { (DEV)->trace_ctx, __FUNCTION__, (type_), (grid_dim_), (block_dim_), &args__ }

#define DSC_INSERT_NAMED_TRACE(DEV, T, type_, name_, grid_dim_, block_dim_) \
    dsc_gpu_trace_tracker<T> trace__ { (DEV)->trace_ctx, (name_), (type_), (grid_dim_), (block_dim_), &args__ }


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
                          const T *args) {
        using namespace internal::tracing;

        if (dsc_tracing_is_enabled()) {
            check_if_full<dsc_gpu_trace>(ctx);
            trace_ = next_empty_trace<dsc_gpu_trace>(ctx);
            fill_trace(&trace_->base, name, type, args);
            DSC_GPU_CHECK(gpu_event_create(&trace_->start_event));
            DSC_GPU_CHECK(gpu_event_create(&trace_->stop_event));
            trace_->grid_dim = grid_dim;
            trace_->block_dim = block_dim;
            trace_->elapsed_ms = 0.f;
            DSC_GPU_CHECK(gpu_event_record(trace_->start_event));
        }
    }

    ~dsc_gpu_trace_tracker() {
        if (trace_) {
            DSC_GPU_CHECK(gpu_event_record(trace_->stop_event));
        }
    }

private:
    dsc_gpu_trace *trace_ = nullptr;
};

static DSC_INLINE dsc_trace_ctx *dsc_gpu_tracing_init() {
    return internal::tracing::init<dsc_gpu_trace>();
}

static DSC_INLINE void dsc_gpu_tracing_dispose(const dsc_trace_ctx *ctx) {
    internal::tracing::dispose(ctx);
}

static DSC_INLINE void dsc_gpu_tracing_dump(void *trace, FILE *json_file,
                                            const bool to_console, const bool to_json) {
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

    if (to_console) {
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
        printf("*** [%ld] \033[38;5;208m%-12s\033[0m %-40s %.2fms (%6ldus)\t|\t%10.2fGB/s (%ldB)\n",
               base->ingestion_time_us,
               "GPU",
               formatted_kernel_name,
               elapsed_ms,
               elapsed_us,
               bandwidth,
               base->rw_bytes);
    }

    if (to_json) {
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
    }

    if (gpu_trace->to_eval()) {
        DSC_GPU_CHECK(gpu_event_destroy(gpu_trace->start_event));
        DSC_GPU_CHECK(gpu_event_destroy(gpu_trace->stop_event));
    }
}

static DSC_INLINE void dsc_gpu_next_trace(dsc_trace_ctx *ctx) {
    internal::tracing::advance_current_trace<dsc_gpu_trace>(ctx);
}

static DSC_INLINE void dsc_gpu_dump_json_metadata(FILE *json_file, void *extra_info) {
    const dsc_gpu_dev_info *dev_info = (dsc_gpu_dev_info *) extra_info;
    fprintf(json_file, R"({"name":"process_name","ph":"M","pid":%d,"tid":0,"args":{"name":"%s"},"process_sort_index":100})" ",\n",
            dev_info->dev_idx,
            dev_info->name);
    fprintf(json_file, R"({"name":"thread_name","ph":"M","pid":%d,"tid":0,"args":{"name":"Stream"},"thread_sort_index":101})" ",\n",
            dev_info->dev_idx);
}

#else

static DSC_INLINE dsc_trace_ctx *dsc_gpu_tracing_init() { return nullptr; }
static DSC_INLINE void dsc_gpu_tracing_dispose(const dsc_trace_ctx *) {}
static DSC_INLINE void dsc_gpu_tracing_dump(void *, FILE *, bool, bool) {}
static DSC_INLINE void dsc_gpu_next_trace(dsc_trace_ctx *) {}
static DSC_INLINE void dsc_gpu_dump_json_metadata(FILE *, void *) {}

#endif // DSC_TRACING