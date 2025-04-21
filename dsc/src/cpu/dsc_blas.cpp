// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#include "cpu/dsc_blas.h"

#if defined(__AVX2__)
#   include <immintrin.h>

#   define dsc_prefetch(ADDR)  _mm_prefetch((ADDR), _MM_HINT_T0)
#else
#   error "DSC requires AVX2 support"
#endif


#define rank1_12x8_avx2_f32(A, B, idx)                                         \
    do {                                                                       \
        const __m256 beta_p = _mm256_loadu_ps(&(B)[(idx) * param<f32, NR>()]); \
                                                                               \
        /* Broadcast alpha_0 */                                                \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * param<f32, MR>() + 0]);    \
        gamma_0 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_0);                  \
                                                                               \
        /* Broadcast alpha_1 */                                                \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * param<f32, MR>() + 1]);    \
        gamma_1 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_1);                  \
                                                                               \
        /* Broadcast alpha_2 */                                                \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * param<f32, MR>() + 2]);    \
        gamma_2 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_2);                  \
                                                                               \
        /* Broadcast alpha_3 */                                                \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * param<f32, MR>() + 3]);    \
        gamma_3 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_3);                  \
                                                                               \
        /* Broadcast alpha_4 */                                                \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * param<f32, MR>() + 4]);    \
        gamma_4 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_4);                  \
                                                                               \
        /* Broadcast alpha_5 */                                                \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * param<f32, MR>() + 5]);    \
        gamma_5 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_5);                  \
                                                                               \
        /* Broadcast alpha_6 */                                                \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * param<f32, MR>() + 6]);    \
        gamma_6 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_6);                  \
                                                                               \
        /* Broadcast alpha_7 */                                                \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * param<f32, MR>() + 7]);    \
        gamma_7 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_7);                  \
                                                                               \
        /* Broadcast alpha_8 */                                                \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * param<f32, MR>() + 8]);    \
        gamma_8 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_8);                  \
                                                                               \
        /* Broadcast alpha_9 */                                                \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * param<f32, MR>() + 9]);    \
        gamma_9 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_9);                  \
                                                                               \
        /* Broadcast alpha_10 */                                               \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * param<f32, MR>() + 10]);   \
        gamma_10 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_10);                \
                                                                               \
        /* Broadcast alpha_11 */                                               \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * param<f32, MR>() + 11]);   \
        gamma_11 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_11);                \
    } while (0)


#define rank1_12x4_avx2_f64(A, B, idx)                                          \
    do {                                                                        \
        const __m256d beta_p = _mm256_loadu_pd(&(B)[(idx) * param<f64, NR>()]); \
                                                                                \
        /* Broadcast alpha_0 */                                                 \
        alpha_pj = _mm256_broadcast_sd(&(A)[(idx) * param<f64, MR>() + 0]);     \
        gamma_0 = _mm256_fmadd_pd(alpha_pj, beta_p, gamma_0);                   \
                                                                                \
        /* Broadcast alpha_1 */                                                 \
        alpha_pj = _mm256_broadcast_sd(&(A)[(idx) * param<f64, MR>() + 1]);     \
        gamma_1 = _mm256_fmadd_pd(alpha_pj, beta_p, gamma_1);                   \
                                                                                \
        /* Broadcast alpha_2 */                                                 \
        alpha_pj = _mm256_broadcast_sd(&(A)[(idx) * param<f64, MR>() + 2]);     \
        gamma_2 = _mm256_fmadd_pd(alpha_pj, beta_p, gamma_2);                   \
                                                                                \
        /* Broadcast alpha_3 */                                                 \
        alpha_pj = _mm256_broadcast_sd(&(A)[(idx) * param<f64, MR>() + 3]);     \
        gamma_3 = _mm256_fmadd_pd(alpha_pj, beta_p, gamma_3);                   \
                                                                                \
        /* Broadcast alpha_4 */                                                 \
        alpha_pj = _mm256_broadcast_sd(&(A)[(idx) * param<f64, MR>() + 4]);     \
        gamma_4 = _mm256_fmadd_pd(alpha_pj, beta_p, gamma_4);                   \
                                                                                \
        /* Broadcast alpha_5 */                                                 \
        alpha_pj = _mm256_broadcast_sd(&(A)[(idx) * param<f64, MR>() + 5]);     \
        gamma_5 = _mm256_fmadd_pd(alpha_pj, beta_p, gamma_5);                   \
                                                                                \
        /* Broadcast alpha_6 */                                                 \
        alpha_pj = _mm256_broadcast_sd(&(A)[(idx) * param<f64, MR>() + 6]);     \
        gamma_6 = _mm256_fmadd_pd(alpha_pj, beta_p, gamma_6);                   \
                                                                                \
        /* Broadcast alpha_7 */                                                 \
        alpha_pj = _mm256_broadcast_sd(&(A)[(idx) * param<f64, MR>() + 7]);     \
        gamma_7 = _mm256_fmadd_pd(alpha_pj, beta_p, gamma_7);                   \
                                                                                \
        /* Broadcast alpha_8 */                                                 \
        alpha_pj = _mm256_broadcast_sd(&(A)[(idx) * param<f64, MR>() + 8]);     \
        gamma_8 = _mm256_fmadd_pd(alpha_pj, beta_p, gamma_8);                   \
                                                                                \
        /* Broadcast alpha_9 */                                                 \
        alpha_pj = _mm256_broadcast_sd(&(A)[(idx) * param<f64, MR>() + 9]);     \
        gamma_9 = _mm256_fmadd_pd(alpha_pj, beta_p, gamma_9);                   \
                                                                                \
        /* Broadcast alpha_10 */                                                \
        alpha_pj = _mm256_broadcast_sd(&(A)[(idx) * param<f64, MR>() + 10]);    \
        gamma_10 = _mm256_fmadd_pd(alpha_pj, beta_p, gamma_10);                 \
                                                                                \
        /* Broadcast alpha_11 */                                                \
        alpha_pj = _mm256_broadcast_sd(&(A)[(idx) * param<f64, MR>() + 11]);    \
        gamma_11 = _mm256_fmadd_pd(alpha_pj, beta_p, gamma_11);                 \
    } while (0)


struct dsc_blas_ctx {
    f64 *packed_a_f64, *packed_b_f64;
    f32 *packed_a_f32, *packed_b_f32;
};

enum gemm_param : u8 {
    MC,
    NC,
    KC,
    MR,
    NR
};

template<typename T, gemm_param P>
consteval int param() {
    static_assert(dsc_is_real<T>(), "param - T must be real");

    if (dsc_is_type<T, f32>()) {
        if (P == MC) return 132;
        if (P == NC) return 2824;
        if (P == KC) return 384;
        if (P == MR) return 12;
        if (P == NR) return 8;
    } else {
        if (P == MC) return 132;
        if (P == NC) return 2824;
        if (P == KC) return 384;
        if (P == MR) return 12;
        if (P == NR) return 4;
    }
}


// ============================================================
// Setup / Teardown
//

dsc_blas_ctx *dsc_blas_init() {
    DSC_LOG_INFO("initializing BLAS context");
    dsc_blas_ctx *ctx = (dsc_blas_ctx *) malloc(sizeof(dsc_blas_ctx));

    ctx->packed_a_f64 = (f64 *) calloc(param<f64, MC>() * param<f64, MC>(), sizeof(f64));
    ctx->packed_a_f32 = (f32 *) calloc(param<f32, MC>() * param<f32, MC>(), sizeof(f32));
    ctx->packed_b_f64 = (f64 *) calloc(param<f64, NC>() * param<f64, KC>(), sizeof(f64));
    ctx->packed_b_f32 = (f32 *) calloc(param<f32, NC>() * param<f32, KC>(), sizeof(f32));

    return ctx;
}

void dsc_blas_destroy(dsc_blas_ctx *ctx) {
    free(ctx->packed_a_f64), free(ctx->packed_a_f32);
    free(ctx->packed_b_f64), free(ctx->packed_b_f32);

    free(ctx);
    DSC_LOG_INFO("BLAS context disposed");
}

// ============================================================
// Matmul operations
//

template<typename T>
static DSC_INLINE void pack_a(const int m, const int k,
                              const T *DSC_RESTRICT a, const int stride_a,
                              T *DSC_RESTRICT packed_a) {
    static constexpr int Mr = param<T, MR>();

    for (int i = 0; i < m; i += Mr) {
        const int ib = DSC_MIN(m - i, Mr);

        for (int p = 0; p < k; ++p) {
            for (int ii = 0; ii < ib; ++ii) *packed_a++ = a[(i + ii) * stride_a + p];
            for (int ii = ib; ii < Mr; ++ii) *packed_a++ = dsc_zero<T>();
        }
    }
}

template<typename T, bool trans_b>
static DSC_INLINE void pack_b(const int k, const int n,
                              const T *DSC_RESTRICT b, const int stride_b,
                              T *DSC_RESTRICT packed_b) {
    static constexpr int Nr = param<T, NR>();

    for (int j = 0; j < n; j += Nr) {
        const int jb = DSC_MIN(n - j, Nr);

        for (int p = 0; p < k; ++p) {
            if constexpr (trans_b) {
                for (int jj = 0; jj < jb; ++jj) *packed_b++ = b[(j + jj) * stride_b + p];
            } else {
                for (int jj = 0; jj < jb; ++jj) *packed_b++ = b[p * stride_b + (j + jj)];
            }
            for (int jj = jb; jj < Nr; ++jj) *packed_b++ = dsc_zero<T>();
        }
    }
}

static DSC_INLINE void ukernel_avx2_f32(const int k,
                                        const f32 *DSC_RESTRICT a,
                                        const f32 *DSC_RESTRICT b,
                                        f32 *DSC_RESTRICT c, const int stride_c) {
    __m256 gamma_0 = _mm256_load_ps(&c[0 * stride_c]);
    __m256 gamma_1 = _mm256_load_ps(&c[1 * stride_c]);
    __m256 gamma_2 = _mm256_load_ps(&c[2 * stride_c]);
    __m256 gamma_3 = _mm256_load_ps(&c[3 * stride_c]);
    __m256 gamma_4 = _mm256_load_ps(&c[4 * stride_c]);
    __m256 gamma_5 = _mm256_load_ps(&c[5 * stride_c]);
    __m256 gamma_6 = _mm256_load_ps(&c[6 * stride_c]);
    __m256 gamma_7 = _mm256_load_ps(&c[7 * stride_c]);
    __m256 gamma_8 = _mm256_load_ps(&c[8 * stride_c]);
    __m256 gamma_9 = _mm256_load_ps(&c[9 * stride_c]);
    __m256 gamma_10 = _mm256_load_ps(&c[10 * stride_c]);
    __m256 gamma_11 = _mm256_load_ps(&c[11 * stride_c]);

    __m256 alpha_pj;

    // Unroll by 4
    const int pb = (k / 4) * 4;
    for (int p = 0; p < pb; p += 4) {
        rank1_12x8_avx2_f32(a, b, p + 0);
        rank1_12x8_avx2_f32(a, b, p + 1);
        rank1_12x8_avx2_f32(a, b, p + 2);
        rank1_12x8_avx2_f32(a, b, p + 3);
    }

    // Scalar case
    for (int p = pb; p < k; ++p) {
        rank1_12x8_avx2_f32(a, b, p);
    }

    _mm256_storeu_ps(&c[0 * stride_c], gamma_0);
    _mm256_storeu_ps(&c[1 * stride_c], gamma_1);
    _mm256_storeu_ps(&c[2 * stride_c], gamma_2);
    _mm256_storeu_ps(&c[3 * stride_c], gamma_3);
    _mm256_storeu_ps(&c[4 * stride_c], gamma_4);
    _mm256_storeu_ps(&c[5 * stride_c], gamma_5);
    _mm256_storeu_ps(&c[6 * stride_c], gamma_6);
    _mm256_storeu_ps(&c[7 * stride_c], gamma_7);
    _mm256_storeu_ps(&c[8 * stride_c], gamma_8);
    _mm256_storeu_ps(&c[9 * stride_c], gamma_9);
    _mm256_storeu_ps(&c[10 * stride_c], gamma_10);
    _mm256_storeu_ps(&c[11 * stride_c], gamma_11);
}

static DSC_INLINE void ukernel_avx2_f64(const int k,
                                        const f64 *DSC_RESTRICT a,
                                        const f64 *DSC_RESTRICT b,
                                        f64 *DSC_RESTRICT c, const int stride_c) {
    __m256d gamma_0 = _mm256_load_pd(&c[0 * stride_c]);
    __m256d gamma_1 = _mm256_load_pd(&c[1 * stride_c]);
    __m256d gamma_2 = _mm256_load_pd(&c[2 * stride_c]);
    __m256d gamma_3 = _mm256_load_pd(&c[3 * stride_c]);
    __m256d gamma_4 = _mm256_load_pd(&c[4 * stride_c]);
    __m256d gamma_5 = _mm256_load_pd(&c[5 * stride_c]);
    __m256d gamma_6 = _mm256_load_pd(&c[6 * stride_c]);
    __m256d gamma_7 = _mm256_load_pd(&c[7 * stride_c]);
    __m256d gamma_8 = _mm256_load_pd(&c[8 * stride_c]);
    __m256d gamma_9 = _mm256_load_pd(&c[9 * stride_c]);
    __m256d gamma_10 = _mm256_load_pd(&c[10 * stride_c]);
    __m256d gamma_11 = _mm256_load_pd(&c[11 * stride_c]);

    __m256d alpha_pj;

    // Unroll by 4
    const int pb = (k / 4) * 4;
    for (int p = 0; p < pb; p += 4) {
        rank1_12x4_avx2_f64(a, b, p + 0);
        rank1_12x4_avx2_f64(a, b, p + 1);
        rank1_12x4_avx2_f64(a, b, p + 2);
        rank1_12x4_avx2_f64(a, b, p + 3);
    }

    // Scalar case
    for (int p = pb; p < k; ++p) {
        rank1_12x4_avx2_f64(a, b, p);
    }

    _mm256_storeu_pd(&c[0 * stride_c], gamma_0);
    _mm256_storeu_pd(&c[1 * stride_c], gamma_1);
    _mm256_storeu_pd(&c[2 * stride_c], gamma_2);
    _mm256_storeu_pd(&c[3 * stride_c], gamma_3);
    _mm256_storeu_pd(&c[4 * stride_c], gamma_4);
    _mm256_storeu_pd(&c[5 * stride_c], gamma_5);
    _mm256_storeu_pd(&c[6 * stride_c], gamma_6);
    _mm256_storeu_pd(&c[7 * stride_c], gamma_7);
    _mm256_storeu_pd(&c[8 * stride_c], gamma_8);
    _mm256_storeu_pd(&c[9 * stride_c], gamma_9);
    _mm256_storeu_pd(&c[10 * stride_c], gamma_10);
    _mm256_storeu_pd(&c[11 * stride_c], gamma_11);
}

void dsc_dgemm(dsc_blas_ctx *ctx, const dsc_blas_trans trans_b,
               const int m, const int n, const int k,
               const f64 *DSC_RESTRICT a, const int stride_a,
               const f64 *DSC_RESTRICT b, const int stride_b,
               f64 *DSC_RESTRICT c, const int stride_c) {
    static constexpr int Nc = param<f64, NC>();
    static constexpr int Mc = param<f64, MC>();
    static constexpr int Kc = param<f64, KC>();
    static constexpr int Mr = param<f64, MR>();
    static constexpr int Nr = param<f64, NR>();

    f64 *DSC_RESTRICT packed_a = ctx->packed_a_f64;
    f64 *DSC_RESTRICT packed_b = ctx->packed_b_f64;


    // 5th loop
    for (int j = 0; j < n; j += Nc) {
        const int jb = DSC_MIN(n - j, Nc);

        // 4th loop
        for (int p = 0; p < k; p += Kc) {
            const int pb = DSC_MIN(k - p, Kc);

            // Pack B
            if (trans_b) {
                pack_b<f64, true>(pb, jb, &b[j * stride_b + p], stride_b, packed_b);
            } else {
                pack_b<f64, false>(pb, jb, &b[p * stride_b + j], stride_b, packed_b);
            }

            // 3rd loop
            for (int i = 0; i < m; i += Mc) {
                const int ib = DSC_MIN(m - i, Mc);

                // Pack A
                pack_a(ib, pb, &a[i * stride_a + p], stride_a, packed_a);

                for (int jj = 0; jj < jb; jj += Nr) {
                    const int jjb = DSC_MIN(jb - jj, Nr);
                    for (int ii = 0; ii < ib; ii += Mr) {
                        const int iib = DSC_MIN(ib - ii, Mr);

                        // Prefetch the current micro-panel of C
                        dsc_prefetch(&c[(i + ii + 0) * stride_c + (j + jj)]);
                        dsc_prefetch(&c[(i + ii + 1) * stride_c + (j + jj)]);
                        dsc_prefetch(&c[(i + ii + 2) * stride_c + (j + jj)]);
                        dsc_prefetch(&c[(i + ii + 3) * stride_c + (j + jj)]);
                        dsc_prefetch(&c[(i + ii + 4) * stride_c + (j + jj)]);
                        dsc_prefetch(&c[(i + ii + 5) * stride_c + (j + jj)]);
                        dsc_prefetch(&c[(i + ii + 6) * stride_c + (j + jj)]);
                        dsc_prefetch(&c[(i + ii + 7) * stride_c + (j + jj)]);
                        dsc_prefetch(&c[(i + ii + 8) * stride_c + (j + jj)]);
                        dsc_prefetch(&c[(i + ii + 9) * stride_c + (j + jj)]);
                        dsc_prefetch(&c[(i + ii + 10) * stride_c + (j + jj)]);
                        dsc_prefetch(&c[(i + ii + 11) * stride_c + (j + jj)]);

                        if (iib == Mr && jjb == Nr) {
                            ukernel_avx2_f64(pb,
                                             &packed_a[ii * pb],
                                             &packed_b[jj * pb],
                                             &c[(i + ii) * stride_c + (j + jj)],
                                             stride_c);
                        } else {
                            alignas(64) f64 packed_c[Mr * Nr]{};
                            ukernel_avx2_f64(pb,
                                             &packed_a[ii * pb],
                                             &packed_b[jj * pb],
                                             packed_c,
                                             Nr);
                            for (int iii = 0; iii < iib; ++iii) {
                                for (int jjj = 0; jjj < jjb; ++jjj) c[(i + ii + iii) * stride_c + (j + jj + jjj)] += packed_c[iii * Nr + jjj];
                            }
                        }
                    }
                }
            }
        }
    }
}

extern void dsc_sgemm(dsc_blas_ctx *ctx, const dsc_blas_trans trans_b,
                      const int m, const int n, const int k,
                      const f32 *DSC_RESTRICT a, const int stride_a,
                      const f32 *DSC_RESTRICT b, const int stride_b,
                      f32 *DSC_RESTRICT c, const int stride_c) {
    static constexpr int Nc = param<f32, NC>();
    static constexpr int Mc = param<f32, MC>();
    static constexpr int Kc = param<f32, KC>();
    static constexpr int Mr = param<f32, MR>();
    static constexpr int Nr = param<f32, NR>();

    f32 *DSC_RESTRICT packed_a = ctx->packed_a_f32;
    f32 *DSC_RESTRICT packed_b = ctx->packed_b_f32;
    // 5th loop
    for (int j = 0; j < n; j += Nc) {
        const int jb = DSC_MIN(n - j, Nc);

        // 4th loop
        for (int p = 0; p < k; p += Kc) {
            const int pb = DSC_MIN(k - p, Kc);

            // Pack B
            if (trans_b) {
                pack_b<f32, true>(pb, jb, &b[j * stride_b + p], stride_b, packed_b);
            } else {
                pack_b<f32, false>(pb, jb, &b[p * stride_b + j], stride_b, packed_b);
            }

            // 3rd loop
            for (int i = 0; i < m; i += Mc) {
                const int ib = DSC_MIN(m - i, Mc);

                // Pack A
                pack_a(ib, pb, &a[i * stride_a + p], stride_a, packed_a);

                for (int jj = 0; jj < jb; jj += Nr) {
                    const int jjb = DSC_MIN(jb - jj, Nr);
                    for (int ii = 0; ii < ib; ii += Mr) {
                        const int iib = DSC_MIN(ib - ii, Mr);

                        // Prefetch the current micro-panel of C
                        dsc_prefetch(&c[(i + ii + 0) * stride_c + (j + jj)]);
                        dsc_prefetch(&c[(i + ii + 1) * stride_c + (j + jj)]);
                        dsc_prefetch(&c[(i + ii + 2) * stride_c + (j + jj)]);
                        dsc_prefetch(&c[(i + ii + 3) * stride_c + (j + jj)]);
                        dsc_prefetch(&c[(i + ii + 4) * stride_c + (j + jj)]);
                        dsc_prefetch(&c[(i + ii + 5) * stride_c + (j + jj)]);
                        dsc_prefetch(&c[(i + ii + 6) * stride_c + (j + jj)]);
                        dsc_prefetch(&c[(i + ii + 7) * stride_c + (j + jj)]);
                        dsc_prefetch(&c[(i + ii + 8) * stride_c + (j + jj)]);
                        dsc_prefetch(&c[(i + ii + 9) * stride_c + (j + jj)]);
                        dsc_prefetch(&c[(i + ii + 10) * stride_c + (j + jj)]);
                        dsc_prefetch(&c[(i + ii + 11) * stride_c + (j + jj)]);

                        if (iib == Mr && jjb == Nr) {
                            ukernel_avx2_f32(pb,
                                             &packed_a[ii * pb],
                                             &packed_b[jj * pb],
                                             &c[(i + ii) * stride_c + (j + jj)],
                                             stride_c);
                        } else {
                            alignas(32) f32 packed_c[Mr * Nr]{};
                            ukernel_avx2_f32(pb,
                                             &packed_a[ii * pb],
                                             &packed_b[jj * pb],
                                             packed_c,
                                             Nr);
                            for (int iii = 0; iii < iib; ++iii) {
                                for (int jjj = 0; jjj < jjb; ++jjj) c[(i + ii + iii) * stride_c + (j + jj + jjj)] += packed_c[iii * Nr + jjj];
                            }
                        }
                    }
                }
            }
        }
    }
}

extern void dsc_dgevm_trans(dsc_blas_ctx *,
                            const int n, const int k,
                            const f64 *DSC_RESTRICT a,
                            const f64 *DSC_RESTRICT b, const int stride_b,
                            f64 *DSC_RESTRICT c) {
    for (int j = 0; j < n; ++j) {
        for (int p = 0; p < k; ++p) {
            c[j] += a[p] * b[j * stride_b + p];
        }
    }
}

extern void dsc_sgevm_trans(dsc_blas_ctx *,
                            const int n, const int k,
                            const f32 *DSC_RESTRICT a,
                            const f32 *DSC_RESTRICT b, const int stride_b,
                            f32 *DSC_RESTRICT c) {
    for (int j = 0; j < n; ++j) {
        for (int p = 0; p < k; ++p) {
            c[j] += a[p] * b[j * stride_b + p];
        }
    }
}