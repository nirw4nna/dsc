// Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include <cublas_v2.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#include <curand_kernel.h>
#pragma GCC diagnostic pop

#define DSC_GPU_PLATFORM CUDA

#define DSC_GPU_CHECK(err)                                          \
    do {                                                            \
        if (err != cudaSuccess) {                                   \
            DSC_LOG_FATAL("CUDA error: %s", cudaGetErrorName(err)); \
        }                                                           \
    } while (0)

#define DSC_GPU_BLAS_CHECK(err)                                            \
    do {                                                                   \
        if (err != CUBLAS_STATUS_SUCCESS) {                                \
            DSC_LOG_FATAL("cuBLAS error: %s", cublasGetStatusString(err)); \
        }                                                                  \
    } while (0)


// ============================================================
// Runtime API
//

#define gpu_get_device_count        cudaGetDeviceCount
#define gpu_get_device_properties   cudaGetDeviceProperties
#define gpu_device_sync             cudaDeviceSynchronize

#define gpu_malloc      cudaMalloc
#define gpu_free        cudaFree
#define gpu_memcpy      cudaMemcpy
#define gpu_memset      cudaMemset
#define gpu_set_device  cudaSetDevice

#define gpu_memcpy_default          cudaMemcpyDefault
#define gpu_memcpy_device_2_host    cudaMemcpyDeviceToHost
#define gpu_memcpy_host_2_device    cudaMemcpyHostToDevice
#define gpu_memcpy_device_2_device  cudaMemcpyDeviceToDevice

using gpu_memcpy_kind = cudaMemcpyKind;
using gpu_device_props = cudaDeviceProp;

// ============================================================
// Rand API
//

#define gpu_init_rand       curand_init
#define gpu_sample_normalf  curand_normal
#define gpu_sample_normal   curand_normal_double

using gpu_rand_state = curandState;

// ============================================================
// BLAS API
//

#define gpu_blas_create     cublasCreate
#define gpu_blas_destroy    cublasDestroy
#define gpu_blas_sgemm      cublasSgemm
#define gpu_blas_dgemm      cublasDgemm
#define gpu_blas_op         cublasOperation_t
#define gpu_blas_dtype      cudaDataType
#define GPU_BLAS_OP_T       CUBLAS_OP_T
#define GPU_BLAS_OP_N       CUBLAS_OP_N
#define GPU_GEMM_DTYPE_BF16 CUDA_R_16BF
#define GPU_GEMM_DTYPE_F32  CUDA_R_32F

using gpu_blas_handle = cublasHandle_t;


static DSC_INLINE cublasStatus_t gpu_blas_bfgemm(const gpu_blas_handle handle, const gpu_blas_op a_op, const gpu_blas_op b_op,
                                                 const int m, const int n, const int k, const void *DSC_RESTRICT alpha,
                                                 const void *DSC_RESTRICT xa, const gpu_blas_dtype a_dtype, const int stride_a,
                                                 const void *DSC_RESTRICT xb, const gpu_blas_dtype b_dtype, const int stride_b,
                                                 const void *DSC_RESTRICT beta, void *out, const gpu_blas_dtype out_dtype,
                                                 const int stride_out, const gpu_blas_dtype compute_dtype) {
    return cublasGemmEx(handle, a_op, b_op, m, n, k,
                        alpha, xa, a_dtype, stride_a,
                        xb, b_dtype, stride_b, beta,
                        out, out_dtype, stride_out,
                        compute_dtype, CUBLAS_GEMM_DEFAULT);
}