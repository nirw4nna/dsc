// Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "dsc_backend.h"

enum dsc_allocator_type : u8 {
    GENERAL_PURPOSE,
    LINEAR
};

static constexpr const char *DSC_ALLOCATOR_NAMES[2] = {
        "General Purpose",
        "Linear"
};

struct dsc_allocator {
    dsc_buffer *buf;
    dsc_allocator_type type;
    void *  (*alloc)        (dsc_buffer *buf, usize nb, usize alignment)    noexcept;
    void    (*clear_buffer) (dsc_buffer *buf)                               noexcept;
    void    (*free)         (dsc_buffer *buf, void *ptr)                    noexcept;
    usize   (*used_memory)  (dsc_buffer *buf)                               noexcept;
};

// ============================================================
// Utilities

extern DSC_MALLOC void *dsc_obj_alloc(dsc_allocator *allocator,
                                      usize nb,
                                      usize alignment = 1) noexcept;

extern void dsc_obj_free(dsc_allocator *allocator, void *ptr) noexcept;

extern void dsc_clear_buffer(dsc_allocator *allocator) noexcept;

extern usize dsc_buffer_used_mem(dsc_allocator *allocator) noexcept;

// ============================================================
// General Purpose Allocator

extern dsc_allocator *dsc_generic_allocator(dsc_buffer *buf) noexcept;

// ============================================================
// Linear (Arena) Allocator

extern dsc_allocator *dsc_linear_allocator(dsc_buffer *buf) noexcept;
