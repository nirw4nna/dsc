// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "dsc.h"

enum dsc_blas_trans : bool {
    NO_TRANS,
    TRANS
};

struct dsc_blas_ctx;

// ============================================================
// Setup / Teardown
//

extern dsc_blas_ctx *dsc_blas_init();

extern void dsc_blas_destroy(dsc_blas_ctx *ctx);

// ============================================================
// GEMM-related functions
//

extern void dsc_dgemm(dsc_blas_ctx *ctx, dsc_blas_trans trans_b,
                      int m, int n, int k,
                      const f64 *DSC_RESTRICT a, int stride_a,
                      const f64 *DSC_RESTRICT b, int stride_b,
                      f64 *DSC_RESTRICT c, int stride_c);

extern void dsc_sgemm(dsc_blas_ctx *ctx, dsc_blas_trans trans_b,
                      int m, int n, int k,
                      const f32 *DSC_RESTRICT a, int stride_a,
                      const f32 *DSC_RESTRICT b, int stride_b,
                      f32 *DSC_RESTRICT c, int stride_c);

// ============================================================
// GEVM-related functions
//

extern void dsc_dgevm_trans(dsc_blas_ctx *ctx,
                            int n, int k,
                            const f64 *DSC_RESTRICT a,
                            const f64 *DSC_RESTRICT b, int stride_b,
                            f64 *DSC_RESTRICT c);

extern void dsc_sgevm_trans(dsc_blas_ctx *ctx,
                            int n, int k,
                            const f32 *DSC_RESTRICT a,
                            const f32 *DSC_RESTRICT b, int stride_b,
                            f32 *DSC_RESTRICT c);