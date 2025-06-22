// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#include "gpu/dsc_gpu.h"
#include "dsc_device.h"

// As per https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses
// "Any address of a variable residing in global memory or returned by one of the memory allocation routines
// from the driver or runtime API is always aligned to at least 256 bytes."
#define DSC_DEVICE_GPU_ALIGN    ((usize) 256)
#define DSC_MEMCPY_DIRECTIONS   ((int) 4)

static constexpr gpu_memcpy_kind DSC_GPU_MEMCPY_DIRECTIONS[DSC_MEMCPY_DIRECTIONS] = {
    gpu_memcpy_default,
    gpu_memcpy_device_2_host,
    gpu_memcpy_host_2_device,
    gpu_memcpy_device_2_device,
};

static DSC_GPU_KERNEL void k_init_random(gpu_rand_state *state) {
    DSC_GPU_TID();
    gpu_init_rand(clock64(), tid, 0, &state[tid]);
}

static void gpu_memcpy_wrapper(void *dst, const void *src, const usize nb, const dsc_memcpy_dir dir) {
    DSC_GPU_CHECK(gpu_memcpy(dst, src, nb, DSC_GPU_MEMCPY_DIRECTIONS[dir]));
}

static void gpu_memset_wrapper(void *dst, const int c, const usize nb) {
    DSC_GPU_CHECK(gpu_memset(dst, c, nb));
}

static void gpu_dispose(dsc_device *dev) {
    DSC_GPU_CHECK(gpu_free(dev->device_mem));

    const dsc_gpu_dev_info *info = (dsc_gpu_dev_info *) dev->extra_info;
    DSC_GPU_BLAS_CHECK(gpu_blas_destroy(info->blas_handle));

    DSC_GPU_CHECK(gpu_free(info->rand_state));

    DSC_LOG_INFO("%s:%d device %s disposed",
                 DSC_DEVICE_NAMES[dev->type],
                 info->dev_idx,
                 info->name);
}

dsc_device *dsc_gpu_device(usize mem_size, const int dev_idx) {
    static dsc_gpu_dev_info extra = {
        .name = {},
        .rand_state = {},
        .blas_handle = {},
        .dev_idx = dev_idx,
    };
    DSC_GPU_BLAS_CHECK(gpu_blas_create(&extra.blas_handle));

    // Allocate 90% of the device memory at most (is this too much?)
    const usize max_mem = (usize) (0.9 * (f64) dsc_gpu_dev_mem(dev_idx));
    mem_size = mem_size < max_mem ? mem_size : DSC_ALIGN(max_mem - (DSC_DEVICE_GPU_ALIGN - 1), DSC_DEVICE_GPU_ALIGN);
    static dsc_device dev = {
        .used_nodes = {},
        .free_nodes = {},
        .head = {},
        .device_mem = {},
        .alignment = DSC_DEVICE_GPU_ALIGN,
        .extra_info = &extra,
        .mem_size = DSC_ALIGN(mem_size, DSC_DEVICE_GPU_ALIGN),
        .used_mem = 0,
        .type = CUDA, // TODO: FIX! this should be either CUDA/ROCM or simply GPU
        .memcpy = gpu_memcpy_wrapper,
        .memset = gpu_memset_wrapper,
        .dispose = gpu_dispose,
    };

    DSC_GPU_CHECK(gpu_set_device(dev_idx));

    dsc_gpu_dev_name(dev_idx, extra.name);

    DSC_GPU_CHECK(gpu_malloc(&extra.rand_state, DSC_GPU_DEFAULT_THREADS * sizeof(gpu_rand_state)));

    k_init_random<<<1, DSC_GPU_DEFAULT_THREADS>>>(extra.rand_state);

    dsc_gpu_sync();

    DSC_GPU_CHECK(gpu_malloc(&dev.device_mem, dev.mem_size));

    dev.free_nodes[0].size = dev.mem_size;
    dev.free_nodes[0].data = dev.device_mem;
    dev.free_nodes[0].next = nullptr;

    dev.head = &dev.free_nodes[0];

    DSC_LOG_INFO("%s:%d device %s initialized with a buffer of %ldMB (total: %ldMB)",
                 DSC_DEVICE_NAMES[dev.type],
                 dev_idx,
                 extra.name,
                 (usize) DSC_B_TO_MB(dev.mem_size),
                 (usize) DSC_B_TO_MB(dsc_gpu_dev_mem(dev_idx)));

    return &dev;
}