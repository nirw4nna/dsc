// Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#include "dsc_tracing.h"

#if defined(DSC_ENABLE_TRACING)
dsc_trace_ctx *g_trace_ctx = nullptr;

void dsc_init_traces(const usize nb) noexcept {
    if (g_trace_ctx == nullptr) {
        const usize max_traces = (usize) (nb / sizeof(dsc_trace));
        g_trace_ctx = (dsc_trace_ctx *) malloc(sizeof(dsc_trace_ctx));
        g_trace_ctx->traces = (dsc_trace *) malloc(max_traces * sizeof(dsc_trace));
        g_trace_ctx->n_traces = 0;
        g_trace_ctx->max_traces = max_traces;
        g_trace_ctx->record = false;
    }
}

void dsc_free_traces() noexcept {
    if (g_trace_ctx != nullptr) {
        free(g_trace_ctx->traces);
        free(g_trace_ctx);
        g_trace_ctx = nullptr;
    }
}
#else
void dsc_init_traces(const usize nb) noexcept {
    DSC_UNUSED(nb);
}

void dsc_free_traces() noexcept {}
#endif // DSC_ENABLE_TRACING