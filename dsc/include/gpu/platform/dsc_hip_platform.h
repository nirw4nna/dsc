// Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include <hip/hip_runtime.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#include <rocrand/rocrand_kernel.h>
#pragma GCC diagnostic pop

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

#define gpu_blas_create         rocblas_create_handle
#define gpu_blas_destroy        rocblas_destroy_handle
#define gpu_blas_sgemm          rocblas_sgemm
#define gpu_blas_dgemm          rocblas_dgemm
#define gpu_blas_op             rocblas_operation
#define gpu_blas_dtype          rocblas_datatype
#define GPU_BLAS_OP_T           rocblas_operation_transpose
#define GPU_BLAS_OP_N           rocblas_operation_none
#define GPU_GEMM_DTYPE_BF16     rocblas_datatype_bf16_r
#define GPU_GEMM_DTYPE_F32      rocblas_datatype_f32_r

using gpu_blas_handle = rocblas_handle;

static DSC_INLINE rocblas_status gpu_blas_bfgemm(const gpu_blas_handle handle, const gpu_blas_op a_op, const gpu_blas_op b_op,
                                                 const int m, const int n, const int k, const void *DSC_RESTRICT alpha,
                                                 const void *DSC_RESTRICT xa, const gpu_blas_dtype a_dtype, const int stride_a,
                                                 const void *DSC_RESTRICT xb, const gpu_blas_dtype b_dtype, const int stride_b,
                                                 const void *DSC_RESTRICT beta, void *out, const gpu_blas_dtype out_dtype,
                                                 const int stride_out, const gpu_blas_dtype compute_dtype) {
    return rocblas_gemm_ex(handle, a_op, b_op, m, n, k,
                           alpha, xa, a_dtype, stride_a,
                           xb, b_dtype, stride_b, beta,
                           out, out_dtype, stride_out,
                           out, out_dtype, stride_out,
                           compute_dtype, rocblas_gemm_algo_standard, 0, 0);
}

// ============================================================
// Event API
//

#define gpu_event_create        hipEventCreate
#define gpu_event_destroy       hipEventDestroy
#define gpu_event_record        hipEventRecord
#define gpu_event_synchronize   hipEventSynchronize
#define gpu_event_elapsed       hipEventElapsedTime

using gpu_event = hipEvent_t;
