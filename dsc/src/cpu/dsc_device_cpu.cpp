// Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#include "dsc_device.h"
#include <cstring>

#if defined(_WIN32)
#   include <malloc.h>

#   define dsc_aligned_alloc(ALIGN, SIZE)   _aligned_malloc(SIZE, ALIGN)
#   define dsc_aligned_free(PTR)            _aligned_free(PTR)
#else
#   include <cstdlib>

#   define dsc_aligned_alloc(ALIGN, SIZE)   aligned_alloc(ALIGN, SIZE)
#   define dsc_aligned_free(PTR)            free(PTR)
#endif

#define DSC_DEVICE_CPU_ALIGN ((usize) 4096)

static void cpu_dispose(dsc_device *dev) {
    dsc_aligned_free(dev->device_mem);

    DSC_LOG_INFO("%s device disposed", DSC_DEVICE_NAMES[dev->type]);
}

static void cpu_memcpy(void *dst, const void *src, const usize nb, dsc_copy_direction) {
    memcpy(dst, src, nb);
}

dsc_device *dsc_cpu_device(const usize mem_size) {
    static dsc_device dev = {
        .mem_size = DSC_ALIGN(mem_size, DSC_DEVICE_CPU_ALIGN),
        .used_mem = 0,
        .type = CPU,
        .dispose = cpu_dispose,
        .memcpy = cpu_memcpy
    };

    dev.device_mem = dsc_aligned_alloc(DSC_DEVICE_CPU_ALIGN, dev.mem_size);
    DSC_ASSERT(dev.device_mem != nullptr);

    dev.free_nodes[0].size = dev.mem_size;
    dev.free_nodes[0].data = dev.device_mem;
    dev.free_nodes[0].next = nullptr;

    DSC_LOG_INFO("%s device initialized with a buffer of %ldMB",
                 DSC_DEVICE_NAMES[dev.type],
                 (usize) DSC_B_TO_MB(dev.mem_size));

    return &dev;
}