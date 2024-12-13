// Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "dsc.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
// Ignore these warnings
#include <curand_kernel.h>
#pragma GCC diagnostic pop


#define DSC_CUDA_KERNEL                     __global__
#define DSC_CUDA_FUNC                       __device__
#define DSC_CUDA_DEFAULT_THREADS_PER_BLOCK  ((int) 256)
#define DSC_CUDA_DEFAULT_BLOCKS             ((int) 256)

#define DSC_CUDA_BLOCKS(n)    (((n) + (DSC_CUDA_DEFAULT_BLOCKS - 1)) / DSC_CUDA_DEFAULT_BLOCKS)
#define DSC_CUDA_THREADS(n)   (((n) + (DSC_CUDA_DEFAULT_THREADS_PER_BLOCK - 1)) / DSC_CUDA_DEFAULT_THREADS_PER_BLOCK)
#define DSC_CUDA_TID()        const size tid = threadIdx.x + blockIdx.x * blockDim.x
#define DSC_CUDA_STRIDE()     const size stride = blockDim.x * gridDim.x

#define DSC_CUDA_FAIL_ON_ERROR(err) \
    do { \
        if (err != cudaSuccess) {                                   \
            DSC_LOG_FATAL("CUDA error: %s", cudaGetErrorName(err)); \
        } \
    } while (0)

struct dsc_device;
struct dsc_cuda_dev_info {
    char name[256];
    curandState *randState;
    int dev_idx;
};

// ============================================================
// Utilities
//

// Enumerate all the available CUDA devices
static DSC_INLINE int dsc_cuda_devices() {
    int devices;
    DSC_CUDA_FAIL_ON_ERROR(cudaGetDeviceCount(&devices));
    return devices;
}

// Return an integer that represents the compute capabilities of this device.
static DSC_INLINE int dsc_cuda_dev_capabilities(const int dev) {
    cudaDeviceProp prop{};
    DSC_CUDA_FAIL_ON_ERROR(cudaGetDeviceProperties(&prop, dev));
    return prop.major * 100 + prop.minor * 10;
}

static DSC_INLINE void dsc_cuda_dev_name(const int dev, char *dst) {
    cudaDeviceProp prop{};
    DSC_CUDA_FAIL_ON_ERROR(cudaGetDeviceProperties(&prop, dev));
    strncpy(dst, prop.name, 256);
}

static DSC_INLINE usize dsc_cuda_dev_mem(const int dev) {
    cudaDeviceProp prop{};
    DSC_CUDA_FAIL_ON_ERROR(cudaGetDeviceProperties(&prop, dev));
    return prop.totalGlobalMem;
}

// ============================================================
// CUDA-specific operations
//

// extern void dsc_cuda_arange(dsc_device *, dsc_tensor *DSC_RESTRICT x);

extern void dsc_cuda_randn(dsc_device *dev, dsc_tensor *DSC_RESTRICT x);
