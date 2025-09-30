// Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "dsc.h"
#include "gpu/dsc_gpu.h"
#include "gpu/platform/dsc_hip_mma.h"

namespace internal::gpu::kernels::flash_attn {

static constexpr int Br_f32 = 32;
static constexpr int Bc_f32 = 64;


static DSC_GPU_KERNEL void k_flash_attention_f32(const f32 *DSC_RESTRICT query,
                                                 const f32 *DSC_RESTRICT key,
                                                 const f32 *DSC_RESTRICT value,
                                                 f32 *DSC_RESTRICT out,
                                                 const bool *DSC_RESTRICT attn_mask,
                                                 const int Nq, const int Nk, const int d,
                                                 const int Tc, const int n_rep,
                                                 const u64 q_stride_batch, const u64 q_stride_head,
                                                 const u64 k_stride_batch, const u64 k_stride_head,
                                                 const u64 v_stride_batch, const u64 v_stride_head,
                                                 const u64 o_stride_batch, const u64 o_stride_head) {
    using namespace internal::gpu;

    const int batch_idx = blockIdx.z;
    const int q_head_idx = blockIdx.y;
    const int kv_head_idx = blockIdx.y / n_rep;
    const int i = blockIdx.x;

    const f32 *DSC_RESTRICT query_block = query + q_stride_batch * batch_idx + q_stride_head * q_head_idx;
    const f32 *DSC_RESTRICT key_block = key + k_stride_batch * batch_idx + k_stride_head * kv_head_idx;
    const f32 *DSC_RESTRICT value_block = value + v_stride_batch * batch_idx + v_stride_head * kv_head_idx;
    f32 *DSC_RESTRICT out_block = out + o_stride_batch * batch_idx + o_stride_head * q_head_idx;

    const int start_row = i * Br_f32;
    const int end_row = DSC_MIN((i + 1) * Br_f32, Nq);
    const int Br_actual = end_row - start_row;
    const int d_eff = (d + 1) & ~1;

    if (Br_actual <= 0) return;

    extern __shared__ f32 shared_mem[];

    f32 *DSC_RESTRICT Qi = shared_mem;                       // Br x d
    f32 *DSC_RESTRICT Kj = Qi + Br_f32 * d_eff;              // Bc x d
    f32 *DSC_RESTRICT Vj = Kj + Bc_f32 * d_eff;              // Bc x d
    f32 *DSC_RESTRICT Sij = Vj + Bc_f32 * d_eff;             // Br x Bc
    f32 *DSC_RESTRICT O_running = Sij + Br_f32 * Bc_f32;     // Br x d
    f32 *DSC_RESTRICT m_running = O_running + Br_f32 * d_eff;// Br
    f32 *DSC_RESTRICT l_running = m_running + Br_f32;        // Br
    f32 *DSC_RESTRICT m_new_row = l_running + Br_f32;        // Br
    f32 *DSC_RESTRICT scaled_old_row = m_new_row + Br_f32;   // Br

    // Thread indices within block
    const int tid = threadIdx.x;
    const int n_threads = blockDim.x;

    // Init running stats
    for (int row_idx = tid; row_idx < Br_f32; row_idx += n_threads) {
        m_running[row_idx] = dsc_inf<f32, false>();
        l_running[row_idx] = 0.f;
    }
    for (int idx = tid; idx < Br_f32 * d_eff; idx += n_threads) {
        O_running[idx] = 0.f;
    }

    // Load Qi in shared memory
    for (int idx = tid; idx < Br_f32 * d_eff; idx += n_threads) {
        const int row = idx / d_eff;
        const int col = idx % d_eff;
        if (row < Br_actual && col < d) {
            Qi[idx] = query_block[(start_row + row) * d + col];
        } else {
            Qi[idx] = 0.f;
        }
    }
    __syncthreads();

    const f32 scaling_factor = 1.f / sqrtf((f32) d);
    // Loop over column block of K and V
    for (int j = 0; j < Tc; ++j) {
        const int start_col = j * Bc_f32;
        const int end_col = DSC_MIN((j + 1) * Bc_f32, Nk);
        const int Bc_actual = end_col - start_col;
        if (Bc_actual <= 0) continue;

        // Load Kj and Vj in shared memory
        for (int idx = tid; idx < Bc_f32 * d_eff; idx += n_threads) {
            const int row = idx / d_eff;
            const int col = idx % d_eff;
            if (row < Bc_actual && col < d) {
                Kj[row * d_eff + col] = key_block[(start_col + row) * d + col];
                Vj[row * d_eff + col] = value_block[(start_col + row) * d + col];
            } else {
                Kj[row * d_eff + col] = 0.f;
                Vj[row * d_eff + col] = 0.f;
            }
        }
        __syncthreads();

        // Compute Sij = Qi * Kj^T
        hip::warp_mm_32x64xk_f32_T(Qi, d_eff, Kj, d_eff,
                                   Sij, Bc_f32, Br_actual, Bc_actual,
                                   d_eff, scaling_factor);

        if (attn_mask != nullptr) {
            for (int idx = tid; idx < Br_actual * Bc_actual; idx += n_threads) {
                const int row = idx / Bc_actual;
                const int col = idx % Bc_actual;
                if (!attn_mask[(row + start_row) * Nk + (col + start_col)]) {
                    Sij[row * Bc_f32 + col] = dsc_inf<f32, false>();
                }
            }
        }

        __syncthreads();

        if (tid < Br_actual) {
            const int row = tid;
            f32 row_max = dsc_inf<f32, false>();

            // Rowmax of Sij
            for (int col = 0; col < Bc_actual; ++col) {
                row_max = fmaxf(row_max, Sij[row * Bc_f32 + col]);
            }
            const f32 m_old = m_running[row];
            const f32 m_new = fmaxf(m_old, row_max);
            m_new_row[row] = m_new;
            // P_tilde_ij
            scaled_old_row[row] = expf(m_old - m_new);
        }
        __syncthreads();

        if (tid < Br_actual) {
            const int row = tid;
            const f32 m_new = m_new_row[row];

            f32 row_sum = 0.f;
            // Compute P_tilde
            for (int col = 0; col < Bc_actual; ++col) {
                const f32 val = expf(Sij[row * Bc_f32 + col] - m_new);
                Sij[row * Bc_f32 + col] = val;
                row_sum += val;
            }

            const f32 l_old = l_running[row];
            l_running[row] = scaled_old_row[row] * l_old + row_sum;
            m_running[row] = m_new;
        }
        __syncthreads();

        hip::warp_mm_32x64xk_f32_beta(Sij, Bc_f32, Vj, d_eff,
                                      O_running, d_eff, Br_actual, d,
                                      Bc_actual, scaled_old_row);
        __syncthreads();
    }

    for (int idx = tid; idx < Br_actual * d; idx += n_threads) {
        const int row = idx / d;
        const int col = idx % d;
        out_block[(start_row + row) * d + col] = O_running[row * d_eff + col] / l_running[row];
    }
}
}
