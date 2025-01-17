// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#include "dsc_device.h"
#include "cuda/dsc_cuda.h"

// As per https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses
// "Any address of a variable residing in global memory or returned by one of the memory allocation routines
// from the driver or runtime API is always aligned to at least 256 bytes."
#define DSC_DEVICE_CUDA_ALIGN ((usize) 256)
#define DSC_MEMCPY_DIRECTIONS ((int) 3)

static constexpr cudaMemcpyKind DSC_CUDA_MEMCPY_DIRECTIONS[DSC_MEMCPY_DIRECTIONS] = {
    cudaMemcpyDeviceToHost,
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToDevice,
};

static DSC_CUDA_KERNEL void k_init_random(curandState *state) {
    DSC_CUDA_TID();
    curand_init(clock64(), tid, 0, &state[tid]);
}

static void cuda_memcpy(void *dst, const void *src, const usize nb, const dsc_memcpy_dir dir) {
    DSC_CUDA_FAIL_ON_ERROR(cudaMemcpy(dst, src, nb, DSC_CUDA_MEMCPY_DIRECTIONS[dir]));
}

static void cuda_dispose(dsc_device *dev) {
    DSC_CUDA_FAIL_ON_ERROR(cudaFree(dev->device_mem));

    const dsc_cuda_dev_info *info = (dsc_cuda_dev_info *) dev->extra_info;

    DSC_CUDA_FAIL_ON_ERROR(cudaFree(info->randState));

    DSC_LOG_INFO("%s:%d device %s disposed",
                 DSC_DEVICE_NAMES[dev->type],
                 info->dev_idx,
                 info->name);
}

dsc_device *dsc_cuda_device(usize mem_size, const int cuda_dev) {
    static dsc_cuda_dev_info extra = {
        .name = {},
        .randState = {},
        .dev_idx = cuda_dev,
    };
    // Allocate 90% of the device memory at most (is this too much?)
    const usize max_mem = (usize) (0.9 * (f64) dsc_cuda_dev_mem(cuda_dev));
    mem_size = mem_size < max_mem ? mem_size : DSC_ALIGN(max_mem - (DSC_DEVICE_CUDA_ALIGN - 1), DSC_DEVICE_CUDA_ALIGN);
    static dsc_device dev = {
        .used_nodes = {},
        .free_nodes = {},
        .head = {},
        .device_mem = {},
        .alignment = DSC_DEVICE_CUDA_ALIGN,
        .extra_info = &extra,
        .mem_size = DSC_ALIGN(mem_size, DSC_DEVICE_CUDA_ALIGN),
        .used_mem = 0,
        .type = CUDA,
        .memcpy = cuda_memcpy,
        .dispose = cuda_dispose,
    };

    DSC_CUDA_FAIL_ON_ERROR(cudaSetDevice(cuda_dev));

    dsc_cuda_dev_name(cuda_dev, extra.name);

    DSC_CUDA_FAIL_ON_ERROR(cudaMalloc(&extra.randState, DSC_CUDA_DEFAULT_THREADS * sizeof(curandState)));

    k_init_random<<<1, DSC_CUDA_DEFAULT_THREADS>>>(extra.randState);

    dsc_cuda_sync();

    DSC_CUDA_FAIL_ON_ERROR(cudaMalloc(&dev.device_mem, dev.mem_size));

    dev.free_nodes[0].size = dev.mem_size;
    dev.free_nodes[0].data = dev.device_mem;
    dev.free_nodes[0].next = nullptr;

    dev.head = &dev.free_nodes[0];

    DSC_LOG_INFO("%s:%d device %s initialized with a buffer of %ldMB (total: %ldMB)",
                 DSC_DEVICE_NAMES[dev.type],
                 cuda_dev,
                 extra.name,
                 (usize) DSC_B_TO_MB(dev.mem_size),
                 (usize) DSC_B_TO_MB(dsc_cuda_dev_mem(cuda_dev)));

    return &dev;
}