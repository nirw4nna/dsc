// Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "dsc.h"

enum dsc_backend_type : u8 {
    CPU = 0,
};

struct dsc_buffer {
    usize size;
    dsc_backend_type backend;
};

static constexpr const char *DSC_BACKED_NAMES[1] = {
        "CPU"
};

struct dsc_backend {
    dsc_buffer *        (*buffer_alloc) (usize nb)          noexcept;
    void                (*buffer_free)  (dsc_buffer *buf)   noexcept;
    dsc_backend_type    (*backend_type) ()                  noexcept;
};

extern dsc_backend_type dsc_get_backend_type(dsc_backend *backend) noexcept;

extern dsc_buffer *dsc_backend_buf_alloc(dsc_backend *backend, usize nb) noexcept;

extern void dsc_backend_buf_free(dsc_backend *backend, dsc_buffer *buf) noexcept;

extern dsc_backend *dsc_cpu_backend() noexcept;


