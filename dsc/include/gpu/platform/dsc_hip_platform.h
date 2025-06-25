// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include <hip/hip_runtime.h>
#include <rocrand/rocrand_kernel.h>
#include <rocblas/rocblas.h>


#define DSC_GPU_PLATFORM ROCM

#define DSC_GPU_CHECK(err)                                          \
    do {                                                            \
        if (err != hipSuccess) {                                    \
            DSC_LOG_FATAL("HIP error: %s", hipGetErrorString(err)); \
        }                                                           \
    } while (0)

#define DSC_GPU_BLAS_CHECK(err)                                                \
    do {                                                                       \
        if (err != rocblas_status_success) {                                   \
            DSC_LOG_FATAL("rocBLAS error: %s", rocblas_status_to_string(err)); \
        }                                                                      \
    } while (0)

// ============================================================
// Runtime API
//

#define gpu_get_device_count        hipGetDeviceCount
#define gpu_get_device_properties   hipGetDeviceProperties
#define gpu_device_sync             hipDeviceSynchronize

#define gpu_malloc      hipMalloc
#define gpu_free        hipFree
#define gpu_memcpy      hipMemcpy
#define gpu_memset      hipMemset
#define gpu_set_device  hipSetDevice

#define gpu_memcpy_default          hipMemcpyDefault
#define gpu_memcpy_device_2_host    hipMemcpyDeviceToHost
#define gpu_memcpy_host_2_device    hipMemcpyHostToDevice
#define gpu_memcpy_device_2_device  hipMemcpyDeviceToDevice

using gpu_memcpy_kind = hipMemcpyKind;
using gpu_device_props = hipDeviceProp_t;

// ============================================================
// Rand API
//

#define gpu_init_rand       rocrand_init
#define gpu_sample_normalf  rocrand_normal
#define gpu_sample_normal   rocrand_normal_double
// Default for cuRAND
using gpu_rand_state = rocrand_state_xorwow;

// ============================================================
// BLAS API
//

#define gpu_blas_create     rocblas_create_handle
#define gpu_blas_destroy    rocblas_destroy_handle
#define gpu_blas_sgemm      rocblas_sgemm
#define gpu_blas_dgemm      rocblas_dgemm
#define GPU_BLAS_OP_T       rocblas_operation_transpose
#define GPU_BLAS_OP_N       rocblas_operation_none

using gpu_blas_handle = rocblas_handle;
