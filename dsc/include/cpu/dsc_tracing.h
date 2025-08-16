// Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once


#include "dsc.h"
#include "dsc_tracing_common.h"
#include <unistd.h>     // getpid()
#include <pthread.h>    // pthread_self()


#undef DSC_INSERT_TYPED_TRACE
#undef DSC_INSERT_NAMED_TRACE

#define DSC_INSERT_TYPED_TRACE(T, type_) \
    dsc_cpu_trace_tracker<T> trace__ { dev->trace_ctx, __FUNCTION__, (type_), &args__ }

#define DSC_INSERT_NAMED_TRACE(T, type_, name_) \
    dsc_cpu_trace_tracker<T> trace__ { dev->trace_ctx, (name_), (type_), &args__ }


struct dsc_cpu_trace {
    dsc_trace_common base;

    u64 tid, start_us, stop_us;
    int pid;
};

template<typename T>
struct dsc_cpu_trace_tracker {
    dsc_cpu_trace_tracker(dsc_trace_ctx *ctx,
                          const char *name,
                          const dsc_trace_type type,
                          const T *args) {
        using namespace internal::tracing;

        check_if_full<dsc_cpu_trace>(ctx);
        trace_ = next_empty_trace<dsc_cpu_trace>(ctx);
        fill_trace(&trace_->base, name, type, args);
        trace_->pid = getpid();
        trace_->tid = pthread_self();
        trace_->start_us = time_us();
    }

    ~dsc_cpu_trace_tracker() {
        using namespace internal::tracing;
        trace_->stop_us= time_us();
    }

private:
    dsc_cpu_trace *trace_;
};

static DSC_INLINE dsc_trace_ctx *dsc_cpu_tracing_init() {
    return internal::tracing::init<dsc_cpu_trace>();
}

static DSC_INLINE void dsc_cpu_tracing_dispose(const dsc_trace_ctx *ctx) {
    internal::tracing::dispose(ctx);
}

static DSC_INLINE  void dsc_cpu_tracing_dump(void *trace, FILE *json_file) {
    static constexpr const char *COLOR_NONE = "\033[0m";
    static constexpr const char *COLOR_CUSTOM = "\033[38;5;51m"; // cyan
    static constexpr const char *COLOR_COPY = "\033[38;5;201m"; // deep magenta

    const dsc_cpu_trace *cpu_trace = (dsc_cpu_trace *) trace;

    const dsc_trace_common *base = &cpu_trace->base;
    const u64 elapsed_us = cpu_trace->stop_us - cpu_trace->start_us;
    const f64 bandwidth = (f64) base->rw_bytes / ((f64) elapsed_us * 1e-6 * DSC_GB(1));

    char device_str[16];
    if (base->type == DSC_COPY_OP) {
        snprintf(device_str, sizeof(device_str), "%s <- %s",
                 DSC_DEVICE_NAMES[base->copy.x.device],
                 DSC_DEVICE_NAMES[base->copy.data_device]);
    } else if (base->type == DSC_TO_OP) {
        snprintf(device_str, sizeof(device_str), "%s <- %s",
                 DSC_DEVICE_NAMES[base->to.new_device],
                 DSC_DEVICE_NAMES[base->to.x.device]);
    } else if (base->type == DSC_GET_IDX) {
        snprintf(device_str, sizeof(device_str), "%s <- %s",
                 DSC_DEVICE_NAMES[base->get_idx.x.device],
                 DSC_DEVICE_NAMES[base->get_idx.x.device]);
    } else if (base->type == DSC_GET_TENSOR) {
        snprintf(device_str, sizeof(device_str), "%s <- %s",
                 DSC_DEVICE_NAMES[base->get_tensor.x.device],
                 DSC_DEVICE_NAMES[base->get_tensor.x.device]);
    } else {
        snprintf(device_str, sizeof(device_str), "CPU");
    }

    const char *ansi_color_1 = "", *ansi_color_2 = "";
    switch (base->type) {
        case DSC_COPY_OP:
        case DSC_TO_OP:
        case DSC_GET_IDX:
        case DSC_GET_TENSOR:
            ansi_color_1 = COLOR_COPY;
            ansi_color_2 = COLOR_NONE;
            break;
        case DSC_TRACE_CUSTOM:
            ansi_color_1 = COLOR_CUSTOM;
            ansi_color_2 = COLOR_NONE;
            break;
        default:
            break;
    }

    printf("*** [%ld] %-12s %s%-40s%s %.2fms (%6ldus)\t|",
           base->ingestion_time_us,
           device_str,
           ansi_color_1,
           base->name,
           ansi_color_2,
           (f64) elapsed_us * 1e-3,
           elapsed_us);

    // Don't show bandwidth for custom traces
    if (base->type != DSC_TRACE_CUSTOM && base->type != DSC_TENSOR_ALLOC && base->type != DSC_TENSOR_FREE) {
        printf("\t%10.2fGB/s (%ldB)",
               bandwidth,
               base->rw_bytes);
    }

    printf("\n");

    fprintf(json_file, R"({"name":"%s","cat":"%s","ph":"X","ts":%ld,"dur":%ld,"pid":%d,"tid":%ld)",
            base->name,
            DSC_TRACE_CATEGORY[base->type],
            base->ingestion_time_us,
            elapsed_us,
            cpu_trace->pid,
            cpu_trace->tid);

    fprintf(json_file, R"(,"args":{)");

    if (base->type != DSC_TRACE_CUSTOM) fprintf(json_file, R"("bandwidth":"%.2fGB/s")", bandwidth);

    internal::tracing::dump_trace_base(json_file, base);
    fprintf(json_file, R"(}})" ",\n");
}

static DSC_INLINE void dsc_cpu_next_trace(dsc_trace_ctx *ctx) {
    internal::tracing::advance_current_trace<dsc_cpu_trace>(ctx);
}

static DSC_INLINE void dsc_cpu_dump_json_metadata(FILE *json_file, void *) {
    fprintf(json_file, R"({"name":"process_name","ph":"M","pid":%d,"tid":%ld,"args":{"name":"CPU"},"process_sort_index":0})" ",\n", getpid(), pthread_self());
    fprintf(json_file, R"({"name":"thread_name","ph":"M","pid":%d,"tid":%ld,"args":{"name":"Main Thread"},"thread_sort_index":1})" ",\n", getpid(), pthread_self());
}

static DSC_INLINE void dsc_cpu_insert_user_trace(dsc_trace_ctx *ctx,
                                                 const char *name,
                                                 const u64 start,
                                                 const u64 duration) {
    dsc_cpu_trace *trace = internal::tracing::next_empty_trace<dsc_cpu_trace>(ctx);
    internal::tracing::fill_trace(&trace->base, name, DSC_TRACE_CUSTOM);
    // For user-generated traces the ingestion time must be inserted manually
    trace->base.ingestion_time_us = start;
    trace->start_us = start;
    trace->stop_us = start + duration;
    // NOTE: maybe it's a good idea to put these kind of events on a separate pid/tid?
    trace->pid = getpid();
    trace->tid = pthread_self();
}