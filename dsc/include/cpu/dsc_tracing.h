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


#define INSERT_TYPED_CPU_TRACE(T, type_) \
    dsc_cpu_trace_tracker<T> trace__ { dev->trace_ctx, __FUNCTION__, (type_), &args__ }

#define DSC_TRACE_CAST_OP(X, OUT)    \
    dsc_cast_args args__{};          \
    DSC_TRACE_SET_TENSOR(X, x);      \
    args__.new_dtype = (OUT)->dtype; \
    INSERT_TYPED_CPU_TRACE(dsc_cast_args, DSC_CAST_OP)

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
                      const T *args) : ctx_(ctx) {
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
    dsc_trace_ctx *ctx_;
    dsc_cpu_trace *trace_;
};

static DSC_INLINE dsc_trace_ctx *dsc_cpu_tracing_init() {
    return internal::tracing::init<dsc_cpu_trace>();
}

static DSC_INLINE void dsc_cpu_tracing_dispose(const dsc_trace_ctx *ctx) {
    internal::tracing::dispose(ctx);
}

static void dsc_cpu_tracing_dump(void *trace, FILE *json_file) {
    const dsc_cpu_trace *cpu_trace = (dsc_cpu_trace *) trace;

    const dsc_trace_common *base = &cpu_trace->base;
    const u64 elapsed_us = cpu_trace->stop_us - cpu_trace->start_us;
    const f64 bandwidth = (f64) base->rw_bytes / ((f64) elapsed_us * 1e-6 * DSC_GB(1));


    printf("*** [%ld] %sCPU%s \t%-40s %.2fms (%6ldus)\t|",
           base->ingestion_time_us,
           base->type == DSC_TRACE_CUSTOM ? "\033[38;5;51m" : "",
           base->type == DSC_TRACE_CUSTOM ? "\033[0m" : "",
           base->name,
           (f64) elapsed_us * 1e-3,
           elapsed_us);

    // Don't show bandwidth for custom traces
    if (base->type != DSC_TRACE_CUSTOM) {
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

static void dsc_cpu_next_trace(dsc_trace_ctx *ctx) {
    internal::tracing::advance_current_trace<dsc_cpu_trace>(ctx);
}

static void dsc_cpu_dump_json_metadata(FILE *json_file, void *) {
    fprintf(json_file, R"({"name":"process_name","ph":"M","pid":%d,"tid":%ld,"args":{"name":"CPU"},"process_sort_index":0})" ",\n", getpid(), pthread_self());
    fprintf(json_file, R"({"name":"thread_name","ph":"M","pid":%d,"tid":%ld,"args":{"name":"Main Thread"},"thread_sort_index":1})" ",\n", getpid(), pthread_self());
}

static void dsc_cpu_insert_user_trace(dsc_trace_ctx *ctx,
                                      const char *name,
                                      const u64 start,
                                      const u64 duration) {
    dsc_cpu_trace *trace = internal::tracing::next_empty_trace<dsc_cpu_trace>(ctx);
    internal::tracing::fill_trace(&trace->base, name, DSC_TRACE_CUSTOM);
    trace->start_us = start;
    trace->stop_us = start + duration;
    // NOTE: maybe it's a good idea to put these kind of events on a separate pid/tid?
    trace->pid = getpid();
    trace->tid = pthread_self();
}