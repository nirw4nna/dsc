// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "dsc.h"

#if defined(DSC_CUDA)

#include <cublas_v2.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#include <curand_kernel.h>
#pragma GCC diagnostic pop


#define DSC_CUDA_KERNEL             __global__
#define DSC_CUDA_FUNC               __device__
#define DSC_CUDA_DEFAULT_THREADS    ((int) 256)
#define DSC_CUDA_MAX_BLOCKS         ((int) 256)

#define DSC_CUDA_BLOCKS(n)    DSC_MIN(DSC_CUDA_MAX_BLOCKS, DSC_CEIL(n, DSC_CUDA_DEFAULT_THREADS))
#define DSC_CUDA_TID()        const int tid = (int) (threadIdx.x + blockIdx.x * blockDim.x)
#define DSC_CUDA_STRIDE()     const int stride = (int) (blockDim.x * gridDim.x)

#define DSC_CUDA_FAIL_ON_ERROR(err)                                 \
    do {                                                            \
        if (err != cudaSuccess) {                                   \
            DSC_LOG_FATAL("CUDA error: %s", cudaGetErrorName(err)); \
        }                                                           \
    } while (0)

#define DSC_CUBLAS_FAIL_ON_ERROR(err)               \
    do {                                            \
        if (err != CUBLAS_STATUS_SUCCESS) {         \
            DSC_LOG_FATAL("cuBLAS error: %d", err); \
        }                                           \
    } while (0)


#define dsc_cuda_copy_from(DST, SRC, nb)    DSC_CUDA_FAIL_ON_ERROR(cudaMemcpy((DST), (SRC), (nb), cudaMemcpyDeviceToHost))
#define dsc_cuda_copy_to(DST, SRC, nb)      DSC_CUDA_FAIL_ON_ERROR(cudaMemcpy((DST), (SRC), (nb), cudaMemcpyHostToDevice))

struct dsc_device;

struct dsc_cuda_dev_info {
    char name[256];
    curandState *randState;
    cublasHandle_t cublas_handle;
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

// ============================================================
// CUDA-specific operations
//

extern void dsc_cuda_cast(dsc_device *, const dsc_tensor *DSC_RESTRICT x,
                          dsc_tensor *DSC_RESTRICT out);

extern void dsc_cuda_arange(dsc_device *, dsc_tensor *DSC_RESTRICT x,
                            f64 start, f64 step);

extern void dsc_cuda_repeat(dsc_device *,
                            const dsc_tensor *DSC_RESTRICT x,
                            dsc_tensor *DSC_RESTRICT out,
                            int repeats, int axis_idx);

extern void dsc_cuda_randn(dsc_device *dev, dsc_tensor *DSC_RESTRICT x);

extern void dsc_cuda_concat(dsc_device *dev,
                            dsc_tensor **to_concat,
                            int tensors,
                            dsc_tensor *DSC_RESTRICT out,
                            int axis_idx);

extern void dsc_cuda_transpose(dsc_device *,
                               const dsc_tensor *DSC_RESTRICT x,
                               dsc_tensor *DSC_RESTRICT out,
                               const int *new_shape,
                               const int *new_stride);

extern void dsc_cuda_tril(dsc_device *,
                          const dsc_tensor *DSC_RESTRICT x,
                          int diagonal,
                          dsc_tensor *DSC_RESTRICT out);

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

extern void dsc_cuda_matmul(dsc_device *,
                            const dsc_tensor *DSC_RESTRICT xa,
                            const dsc_tensor *DSC_RESTRICT xb,
                            bool trans_b,
                            dsc_tensor *DSC_RESTRICT out);

extern void dsc_cuda_compare(dsc_device *,
                             const dsc_tensor *xa,
                             const dsc_tensor *xb,
                             dsc_comparison_op comp,
                             dsc_tensor *out);

extern void dsc_cuda_masked_fill(dsc_device *,
                                 dsc_tensor *DSC_RESTRICT x,
                                 const dsc_tensor *DSC_RESTRICT mask,
                                 f64 value);

extern void dsc_cuda_outer(dsc_device *,
                           const dsc_tensor *DSC_RESTRICT xa,
                           const dsc_tensor *DSC_RESTRICT xb,
                           dsc_tensor *DSC_RESTRICT out);

extern void dsc_cuda_where(dsc_device *,
                           const dsc_tensor *DSC_RESTRICT condition,
                           const dsc_tensor *DSC_RESTRICT input,
                           const dsc_tensor *DSC_RESTRICT other,
                           dsc_tensor *DSC_RESTRICT out);

// ============================================================
// Unary Operations

extern void dsc_cuda_cos(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out);

extern void dsc_cuda_sin(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out);

extern void dsc_cuda_tanh(dsc_device *,
                          const dsc_tensor *DSC_RESTRICT x,
                          dsc_tensor *DSC_RESTRICT out);

extern void dsc_cuda_exp(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out);

extern void dsc_cuda_sqrt(dsc_device *,
                          const dsc_tensor *DSC_RESTRICT x,
                          dsc_tensor *DSC_RESTRICT out);

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

