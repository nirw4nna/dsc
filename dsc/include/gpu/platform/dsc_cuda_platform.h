// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
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
#define gpu_memcpy_host_2_device    cudaMemcpyDeviceToHost
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
#define GPU_BLAS_OP_T       CUBLAS_OP_T
#define GPU_BLAS_OP_N       CUBLAS_OP_N

using gpu_blas_handle = cublasHandle_t;
