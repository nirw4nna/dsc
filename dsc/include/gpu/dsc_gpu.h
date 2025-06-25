// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "dsc.h"

#if defined(DSC_CUDA) && defined(DSC_HIP)
#   error "DSC can't be compiled with both CUDA and HIP support"
#endif


#if defined(DSC_CUDA) || defined(DSC_HIP)

#if defined(DSC_CUDA)
#   include "platform/dsc_cuda_platform.h"
#endif

#if defined(DSC_HIP)
#   include "platform/dsc_hip_platform.h"
#endif

#define DSC_GPU_KERNEL             __global__
#define DSC_GPU_FUNC               __device__
#define DSC_GPU_DEFAULT_THREADS    ((int) 256)
#define DSC_GPU_MAX_BLOCKS         ((int) 256)

#define DSC_GPU_BLOCKS(n)    DSC_MIN(DSC_GPU_MAX_BLOCKS, DSC_CEIL(n, DSC_GPU_DEFAULT_THREADS))
#define DSC_GPU_TID()        const int tid = (int) (threadIdx.x + blockIdx.x * blockDim.x)
#define DSC_GPU_STRIDE()     const int stride = (int) (blockDim.x * gridDim.x)

struct dsc_device;

struct dsc_gpu_dev_info {
    char name[256];
    gpu_rand_state *rand_state;
    gpu_blas_handle blas_handle;
    int dev_idx;
    dsc_gpu_platform platform;
};

// ============================================================
// Utilities
//

static DSC_INLINE int dsc_gpu_devices() {
    int devices;
    DSC_GPU_CHECK(gpu_get_device_count(&devices));
    return devices;
}

static DSC_INLINE int dsc_gpu_dev_capability(const int dev) {
    gpu_device_props prop{};
    DSC_GPU_CHECK(gpu_get_device_properties(&prop, dev));
    return prop.major * 100 + prop.minor * 10;
}

static DSC_INLINE void dsc_gpu_dev_name(const int dev, char *dst) {
    gpu_device_props prop{};
    DSC_GPU_CHECK(gpu_get_device_properties(&prop, dev));
    strncpy(dst, prop.name, 256);
}

static DSC_INLINE usize dsc_gpu_dev_mem(const int dev) {
    gpu_device_props prop{};
    DSC_GPU_CHECK(gpu_get_device_properties(&prop, dev));
    return prop.totalGlobalMem;
}

static DSC_INLINE void dsc_gpu_sync() {
    DSC_GPU_CHECK(gpu_device_sync());
}

// ============================================================
// GPU-specific operations
//

extern void dsc_gpu_cast(dsc_device *, const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out);

extern void dsc_gpu_arange(dsc_device *, dsc_tensor *DSC_RESTRICT x,
                           f64 start, f64 step);

extern void dsc_gpu_repeat(dsc_device *,
                           const dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out,
                           int repeats, int axis_idx);

extern void dsc_gpu_randn(dsc_device *dev, dsc_tensor *DSC_RESTRICT x);

extern void dsc_gpu_concat(dsc_device *dev,
                           dsc_tensor **to_concat,
                           int tensors,
                           dsc_tensor *DSC_RESTRICT out,
                           int axis_idx);

extern void dsc_gpu_transpose(dsc_device *,
                              const dsc_tensor *DSC_RESTRICT x,
                              dsc_tensor *DSC_RESTRICT out,
                              const int *new_shape,
                              const int *new_stride);

extern void dsc_gpu_tril(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         int diagonal,
                         dsc_tensor *DSC_RESTRICT out);

// ============================================================
// Indexing and Slicing

extern void dsc_gpu_get_slice(dsc_device *,
                              const dsc_tensor *DSC_RESTRICT x,
                              dsc_tensor *DSC_RESTRICT out,
                              int n_slices, const dsc_slice *slices,
                              bool whole);

extern void dsc_gpu_set_slice(dsc_device *,
                              dsc_tensor *DSC_RESTRICT xa,
                              bool xa_scalar,
                              const dsc_tensor *DSC_RESTRICT xb,
                              bool xb_scalar,
                              int n_slices,
                              const dsc_slice *slices,
                              bool whole);

// ============================================================
// Binary Operations

extern void dsc_gpu_add(dsc_device *,
                        const dsc_tensor *xa,
                        const dsc_tensor *xb,
                        dsc_tensor *out);

extern void dsc_gpu_sub(dsc_device *,
                        const dsc_tensor *xa,
                        const dsc_tensor *xb,
                        dsc_tensor *out);

extern void dsc_gpu_mul(dsc_device *,
                        const dsc_tensor *xa,
                        const dsc_tensor *xb,
                        dsc_tensor *out);

extern void dsc_gpu_div(dsc_device *,
                        const dsc_tensor *xa,
                        const dsc_tensor *xb,
                        dsc_tensor *out);

extern void dsc_gpu_pow(dsc_device *,
                        const dsc_tensor *xa,
                        const dsc_tensor *xb,
                        dsc_tensor *out);

extern void dsc_gpu_matmul(dsc_device *,
                           const dsc_tensor *DSC_RESTRICT xa,
                           const dsc_tensor *DSC_RESTRICT xb,
                           bool trans_b,
                           dsc_tensor *DSC_RESTRICT out);

extern void dsc_gpu_compare(dsc_device *,
                            const dsc_tensor *xa,
                            const dsc_tensor *xb,
                            dsc_comparison_op comp,
                            dsc_tensor *out);

extern void dsc_gpu_masked_fill(dsc_device *,
                                dsc_tensor *DSC_RESTRICT x,
                                const dsc_tensor *DSC_RESTRICT mask,
                                f64 value);

extern void dsc_gpu_outer(dsc_device *,
                          const dsc_tensor *DSC_RESTRICT xa,
                          const dsc_tensor *DSC_RESTRICT xb,
                          dsc_tensor *DSC_RESTRICT out);

extern void dsc_gpu_where(dsc_device *,
                          const dsc_tensor *DSC_RESTRICT condition,
                          const dsc_tensor *DSC_RESTRICT input,
                          const dsc_tensor *DSC_RESTRICT other,
                          dsc_tensor *DSC_RESTRICT out);

// ============================================================
// Unary Operations

extern void dsc_gpu_cos(dsc_device *,
                        const dsc_tensor *DSC_RESTRICT x,
                        dsc_tensor *DSC_RESTRICT out);

extern void dsc_gpu_sin(dsc_device *,
                        const dsc_tensor *DSC_RESTRICT x,
                        dsc_tensor *DSC_RESTRICT out);

extern void dsc_gpu_tanh(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out);

extern void dsc_gpu_exp(dsc_device *,
                        const dsc_tensor *DSC_RESTRICT x,
                        dsc_tensor *DSC_RESTRICT out);

extern void dsc_gpu_sqrt(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out);

// ============================================================
// Unary Operations Along Axis

extern void dsc_gpu_sum(dsc_device *,
                        const dsc_tensor *DSC_RESTRICT x,
                        dsc_tensor *DSC_RESTRICT out,
                        int axis_idx);

extern void dsc_gpu_min(dsc_device *,
                        const dsc_tensor *DSC_RESTRICT x,
                        dsc_tensor *DSC_RESTRICT out,
                        int axis_idx);

extern void dsc_gpu_max(dsc_device *,
                        const dsc_tensor *DSC_RESTRICT x,
                        dsc_tensor *DSC_RESTRICT out,
                        int axis_idx);

#else

static DSC_INLINE int dsc_gpu_devices() {
    return 0;
}

static DSC_INLINE int dsc_gpu_dev_capability(const int dev) {
    return 0;
}

static DSC_INLINE void dsc_gpu_dev_name(const int dev, char *dst) {
    return 0;
}

static DSC_INLINE usize dsc_gpu_dev_mem(const int dev) {
    return 0;
}

static DSC_INLINE void dsc_gpu_sync() {}

#endif // DSC_CUDA || DSC_HIP