// Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#include "dsc_backend.h"

#if defined(_WIN32)
#   include <malloc.h>

#   define dsc_aligned_alloc(ALIGN, SIZE)   _aligned_malloc(SIZE, ALIGN)
#   define dsc_aligned_free(PTR)            _aligned_free(PTR)
#else
#   include <cstdlib>

#   define dsc_aligned_alloc(ALIGN, SIZE)   aligned_alloc(ALIGN, SIZE)
#   define dsc_aligned_free(PTR)            free(PTR)
#endif

#define DSC_BACKEND_CPU_ALIGN ((usize) 4096)

// ============================================================
// Utilities

dsc_backend_type dsc_get_backend_type(dsc_backend *backend) noexcept {
    return backend->backend_type();
}

dsc_buffer *dsc_backend_buf_alloc(dsc_backend *backend, usize nb) noexcept {
    return backend->buffer_alloc(nb);
}

void dsc_backend_buf_free(dsc_backend *backend, dsc_buffer *buf) noexcept {
    return backend->buffer_free(buf);
}

// ============================================================
// CPU Backend

static DSC_MALLOC dsc_buffer *cpu_buffer_alloc(usize nb) noexcept {
    const usize buffer_size = DSC_ALIGN(nb + sizeof(dsc_buffer), DSC_BACKEND_CPU_ALIGN);

    dsc_buffer *buf = (dsc_buffer *) dsc_aligned_alloc(DSC_BACKEND_CPU_ALIGN, buffer_size);

    DSC_ASSERT(buf != nullptr);

    buf->size = buffer_size - sizeof(dsc_buffer);
    buf->backend = dsc_backend_type::CPU;

    return buf;
}

static void cpu_buffer_free(dsc_buffer *buf) noexcept {
    dsc_aligned_free(buf);
}

static DSC_STRICTLY_PURE dsc_backend_type cpu_backend_type() noexcept {
    return dsc_backend_type::CPU;
}

dsc_backend *dsc_cpu_backend() noexcept {
    static dsc_backend backend = {
        /* .buffer_alloc    = */ cpu_buffer_alloc,
        /* .buffer_free     = */ cpu_buffer_free,
        /* .backend_type    = */ cpu_backend_type,
    };
    return &backend;
}
