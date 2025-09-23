// Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "dsc.h"
#include "gpu/dsc_gpu.h"

namespace internal::gpu::hip {

using f32x16 = __attribute__((__vector_size__(16 * sizeof(f32)))) f32;

DSC_GPU_FUNC DSC_INLINE void warp_mm_32x64xk_f32_T(const f32 *DSC_RESTRICT xa,
                                                   const int lda,
                                                   const f32 *DSC_RESTRICT xb,
                                                   const int ldb,
                                                   f32 *DSC_RESTRICT out,
                                                   const int ldo,
                                                   const int m, const int n,
                                                   const int k, const f32 alpha) {
    // Compute out = alpha * (xa * xb^T) using 2 waves
    // where xa is 32xK and Kx64 with K a multiple of 2
    const int tid = threadIdx.x;
    const int wave_id = tid / 64;

    if (wave_id > 2) return;

    const int tid_in_wave = tid % 64;
    const int row_in_wave = tid_in_wave % 32;
    const int col_in_wave = tid_in_wave / 32;

    // Per-wave start
    const int n_start = wave_id * 32;

    // Each wave handles a 32x32 tile that is stored in registers
    f32x16 acc = {0};

    // K must be a multiple of 2 since out basic operation is a 32x2.
    // The caller must pad the shared memory to account for this
    const int k_eff = (k + 1) & ~ 1;

    int xa_idx = row_in_wave * lda + col_in_wave;
    int xb_idx = (row_in_wave + n_start) * ldb + col_in_wave;

    for (int p = 0; p < k_eff/2; ++p) {
        const f32 reg_a = xa[xa_idx];
        const f32 reg_b = xb[xb_idx];

        acc = __builtin_amdgcn_mfma_f32_32x32x2f32(reg_a, reg_b, acc, 0, 0, 0);

        xa_idx += 2;
        xb_idx += 2;
    }

    // The output is a 32x32 matrix spread across 16 VGPRs, each one with 64 lanes as follows:
    // R03 L031     -> D[0:3]
    // R03 L3264    -> D[4:7]
    // R47 L031     -> D[8:11]
    // R47 L3264    -> D[12:15]
    // R811 L031    -> D[16:19]
    // R811 L3264   -> D[20:23]
    // R1215 L031   -> D[24:27]
    // R1515 L3264  -> D[28:31]

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int out_row = i + 4 * col_in_wave + 8 * j;
            const int out_col = row_in_wave + n_start;

            if (out_row < m && out_col < n) {
                out[out_row * ldo + out_col] = alpha * acc[i + 4 * j];
            }
        }
    }
}

DSC_GPU_FUNC DSC_INLINE void warp_mm_32x64xk_f32_beta(const f32 *DSC_RESTRICT xa,
                                                      const int lda,
                                                      const f32 *DSC_RESTRICT xb,
                                                      const int ldb,
                                                      f32 *DSC_RESTRICT out,
                                                      const int ldo,
                                                      const int m, const int n, const int k,
                                                      const f32 *DSC_RESTRICT beta) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 64;

    if (wave_id > 2) return;

    const int tid_in_wave = tid % 64;
    const int row_in_wave = tid_in_wave % 32;
    const int col_in_wave = tid_in_wave / 32;

    // Per-wave start
    const int n_start = wave_id * 32;

    const int k_eff = (k + 1) & ~ 1;

    int xa_idx = row_in_wave * lda + col_in_wave;
    const int b_col = row_in_wave + n_start;

    f32x16 acc = {0};

    for (int p = 0; p < k_eff / 2; ++p) {
        const f32 reg_a = xa[xa_idx];
        // To visualize this mapping remember we are iterating over the rows of B with the index p
        const int b_row = 2 * p + col_in_wave;
        const f32 reg_b = xb[b_row * ldb + b_col];

        acc = __builtin_amdgcn_mfma_f32_32x32x2f32(reg_a, reg_b, acc, 0, 0, 0);

        xa_idx += 2;
    }

    // Compute out = mm_res + beta * old_out
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int out_row = i + 4 * col_in_wave + 8 * j;
            const int out_col = row_in_wave + n_start;

            if (out_row < m && out_col < n) {
                const int out_idx = out_row * ldo + out_col;
                const f32 old_out = out[out_idx];
                const f32 mm_res = acc[i + 4 * j];
                const f32 this_beta = beta[out_row];
                out[out_idx] = mm_res + this_beta * old_out;
            }
        }
    }
}

}