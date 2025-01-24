// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "dsc.h"

#if defined(DSC_CUDA)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#include <curand_kernel.h>
#pragma GCC diagnostic pop


#define DSC_CUDA_KERNEL             __global__
#define DSC_CUDA_FUNC               __device__
#define DSC_CUDA_DEFAULT_THREADS    ((int) 256)
#define DSC_CUDA_MAX_BLOCKS         ((int) 256)

#define DSC_CUDA_BLOCKS(n)    DSC_MIN(DSC_CUDA_MAX_BLOCKS, (((n) + (DSC_CUDA_DEFAULT_THREADS - 1)) / DSC_CUDA_DEFAULT_THREADS))
#define DSC_CUDA_TID()        const int tid = (int) (threadIdx.x + blockIdx.x * blockDim.x)
#define DSC_CUDA_STRIDE()     const int stride = (int) (blockDim.x * gridDim.x)

#define DSC_CUDA_FAIL_ON_ERROR(err) \
    do { \
        if (err != cudaSuccess) {                                   \
            DSC_LOG_FATAL("CUDA error: %s", cudaGetErrorName(err)); \
        } \
    } while (0)

#define dsc_cuda_copy_from(DST, SRC, nb)    DSC_CUDA_FAIL_ON_ERROR(cudaMemcpy((DST), (SRC), (nb), cudaMemcpyDeviceToHost))
#define dsc_cuda_copy_to(DST, SRC, nb)      DSC_CUDA_FAIL_ON_ERROR(cudaMemcpy((DST), (SRC), (nb), cudaMemcpyHostToDevice))

struct dsc_device;

struct dsc_cuda_dev_info {
    char name[256];
    curandState *randState;
    int dev_idx;
};

// ============================================================
// Utilities
//

static DSC_INLINE int dsc_cuda_devices() {
    int devices;
    DSC_CUDA_FAIL_ON_ERROR(cudaGetDeviceCount(&devices));
    return devices;
}

static DSC_INLINE int dsc_cuda_dev_capability(const int dev) {
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

static DSC_INLINE void dsc_cuda_sync() {
    DSC_CUDA_FAIL_ON_ERROR(cudaDeviceSynchronize());
}

extern void dsc_cuda_cast(dsc_device *, const dsc_tensor *DSC_RESTRICT x,
                          dsc_tensor *DSC_RESTRICT out);

extern void dsc_cuda_arange(dsc_device *, dsc_tensor *DSC_RESTRICT x);

extern void dsc_cuda_randn(dsc_device *dev, dsc_tensor *DSC_RESTRICT x);

extern void dsc_cuda_concat(dsc_device *dev,
                            dsc_tensor **to_concat,
                            int tensors,
                            dsc_tensor *DSC_RESTRICT out,
                            int axis_idx);

// ============================================================
// Indexing and Slicing

extern void dsc_cuda_get_slice(dsc_device *,
                               const dsc_tensor *DSC_RESTRICT x,
                               dsc_tensor *DSC_RESTRICT out,
                               int n_slices, const dsc_slice *slices,
                               bool whole);

extern void dsc_cuda_set_slice(dsc_device *,
                               dsc_tensor *DSC_RESTRICT xa,
                               bool xa_scalar,
                               const dsc_tensor *DSC_RESTRICT xb,
                               bool xb_scalar,
                               int n_slices,
                               const dsc_slice *slices,
                               bool whole);

// ============================================================
// Binary Operations

extern void dsc_cuda_add(dsc_device *,
                         const dsc_tensor *xa,
                         const dsc_tensor *xb,
                         dsc_tensor *out);

extern void dsc_cuda_sub(dsc_device *,
                         const dsc_tensor *xa,
                         const dsc_tensor *xb,
                         dsc_tensor *out);

extern void dsc_cuda_mul(dsc_device *,
                         const dsc_tensor *xa,
                         const dsc_tensor *xb,
                         dsc_tensor *out);

extern void dsc_cuda_div(dsc_device *,
                         const dsc_tensor *xa,
                         const dsc_tensor *xb,
                         dsc_tensor *out);

extern void dsc_cuda_pow(dsc_device *,
                         const dsc_tensor *xa,
                         const dsc_tensor *xb,
                         dsc_tensor *out);

// ============================================================
// Unary Operations

extern void dsc_cuda_cos(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out);

extern void dsc_cuda_sin(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out);

extern void dsc_cuda_sinc(dsc_device *,
                          const dsc_tensor *DSC_RESTRICT x,
                          dsc_tensor *DSC_RESTRICT out);

extern void dsc_cuda_logn(dsc_device *,
                          const dsc_tensor *DSC_RESTRICT x,
                          dsc_tensor *DSC_RESTRICT out);

extern void dsc_cuda_log2(dsc_device *,
                          const dsc_tensor *DSC_RESTRICT x,
                          dsc_tensor *DSC_RESTRICT out);

extern void dsc_cuda_log10(dsc_device *,
                           const dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out);

extern void dsc_cuda_exp(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out);

extern void dsc_cuda_sqrt(dsc_device *,
                          const dsc_tensor *DSC_RESTRICT x,
                          dsc_tensor *DSC_RESTRICT out);

extern void dsc_cuda_i0(dsc_device *,
                        const dsc_tensor *DSC_RESTRICT x,
                        dsc_tensor *DSC_RESTRICT out);

extern void dsc_cuda_abs(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out);

extern void dsc_cuda_angle(dsc_device *,
                           const dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out);

extern void dsc_cuda_conj(dsc_device *,
                          const dsc_tensor *DSC_RESTRICT x,
                          dsc_tensor *DSC_RESTRICT out);

extern void dsc_cuda_real(dsc_device *,
                          const dsc_tensor *DSC_RESTRICT x,
                          dsc_tensor *DSC_RESTRICT out);

extern void dsc_cuda_imag(dsc_device *,
                          const dsc_tensor *DSC_RESTRICT x,
                          dsc_tensor *DSC_RESTRICT out);

extern void dsc_cuda_clip(dsc_device *,
                          const dsc_tensor *DSC_RESTRICT x,
                          dsc_tensor *DSC_RESTRICT out,
                          f64 x_min, f64 x_max);

// ============================================================
// Unary Operations Along Axis

extern void dsc_cuda_sum(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out,
                         int axis_idx);

extern void dsc_cuda_min(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out,
                         int axis_idx);

extern void dsc_cuda_max(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out,
                         int axis_idx);
#else

static DSC_INLINE int dsc_cuda_devices() {
    return 0;
}

static DSC_INLINE int dsc_cuda_dev_capability(int) {
    return 0;
}

static DSC_INLINE void dsc_cuda_dev_name(int, char *) {}

static DSC_INLINE usize dsc_cuda_dev_mem(int) {
    return 0;
}

static DSC_INLINE void dsc_cuda_sync() {}

#endif // DSC_CUDA
