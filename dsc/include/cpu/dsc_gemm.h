// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "dsc.h"

#if defined(__AVX2__)
#   include "cpu/simd/dsc_simd256.h"
#else
#   error "DSC requires at least AVX2 support"
#endif


#define rank1_avx2(A, B, idx)                                              \
    do {                                                                   \
        const vec<T> beta_p = load(&(B)[(idx) * get_gemm_param<T, NR>()]); \
                                                                           \
        /* Broadcast alpha_0 */                                            \
        alpha_pj = broadcast(&(A)[(idx) * get_gemm_param<T, MR>() + 0]);   \
        gamma_0 = fmadd(alpha_pj, beta_p, gamma_0);                        \
                                                                           \
        /* Broadcast alpha_1 */                                            \
        alpha_pj = broadcast(&(A)[(idx) * get_gemm_param<T, MR>() + 1]);   \
        gamma_1 = fmadd(alpha_pj, beta_p, gamma_1);                        \
                                                                           \
        /* Broadcast alpha_2 */                                            \
        alpha_pj = broadcast(&(A)[(idx) * get_gemm_param<T, MR>() + 2]);   \
        gamma_2 = fmadd(alpha_pj, beta_p, gamma_2);                        \
                                                                           \
        /* Broadcast alpha_3 */                                            \
        alpha_pj = broadcast(&(A)[(idx) * get_gemm_param<T, MR>() + 3]);   \
        gamma_3 = fmadd(alpha_pj, beta_p, gamma_3);                        \
                                                                           \
        /* Broadcast alpha_4 */                                            \
        alpha_pj = broadcast(&(A)[(idx) * get_gemm_param<T, MR>() + 4]);   \
        gamma_4 = fmadd(alpha_pj, beta_p, gamma_4);                        \
                                                                           \
        /* Broadcast alpha_5 */                                            \
        alpha_pj = broadcast(&(A)[(idx) * get_gemm_param<T, MR>() + 5]);   \
        gamma_5 = fmadd(alpha_pj, beta_p, gamma_5);                        \
                                                                           \
        /* Broadcast alpha_6 */                                            \
        alpha_pj = broadcast(&(A)[(idx) * get_gemm_param<T, MR>() + 6]);   \
        gamma_6 = fmadd(alpha_pj, beta_p, gamma_6);                        \
                                                                           \
        /* Broadcast alpha_7 */                                            \
        alpha_pj = broadcast(&(A)[(idx) * get_gemm_param<T, MR>() + 7]);   \
        gamma_7 = fmadd(alpha_pj, beta_p, gamma_7);                        \
                                                                           \
        /* Broadcast alpha_8 */                                            \
        alpha_pj = broadcast(&(A)[(idx) * get_gemm_param<T, MR>() + 8]);   \
        gamma_8 = fmadd(alpha_pj, beta_p, gamma_8);                        \
                                                                           \
        /* Broadcast alpha_9 */                                            \
        alpha_pj = broadcast(&(A)[(idx) * get_gemm_param<T, MR>() + 9]);   \
        gamma_9 = fmadd(alpha_pj, beta_p, gamma_9);                        \
                                                                           \
        /* Broadcast alpha_10 */                                           \
        alpha_pj = broadcast(&(A)[(idx) * get_gemm_param<T, MR>() + 10]);  \
        gamma_10 = fmadd(alpha_pj, beta_p, gamma_10);                      \
                                                                           \
        /* Broadcast alpha_11 */                                           \
        alpha_pj = broadcast(&(A)[(idx) * get_gemm_param<T, MR>() + 11]);  \
        gamma_11 = fmadd(alpha_pj, beta_p, gamma_11);                      \
    } while (0)

namespace internal::gemm {
enum gemm_param : u8 {
    MC,
    NC,
    KC,
    MR,
    NR
};

template<typename T, gemm_param P>
consteval int get_gemm_param() {
    static_assert(dsc_is_real<T>(), "get_gemm_param - T must be real");

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

template<typename T>
DSC_INLINE void pack_a(const int m, const int k,
                      const T *DSC_RESTRICT a, const int stride_a,
                      T *DSC_RESTRICT packed_a) {
    const int Mr = get_gemm_param<T, MR>();

    for (int i = 0; i < m; i += Mr) {
        const int ib = DSC_MIN(m - i, Mr);

        for (int p = 0; p < k; ++p) {
            for (int ii = 0; ii < ib; ++ii) *packed_a++ = a[(i + ii) * stride_a + p];
            for (int ii = ib; ii < Mr; ++ii) *packed_a++ = dsc_zero<T>();
        }
    }
}

template<typename T, bool trans_b>
DSC_INLINE void pack_b(const int k, const int n,
                      const T *DSC_RESTRICT b, const int stride_b,
                      T *DSC_RESTRICT packed_b) {
    const int Nr = get_gemm_param<T, NR>();

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

template<typename T>
DSC_INLINE void ukernel_avx2(const int k,
                             const T *DSC_RESTRICT a,
                             const T *DSC_RESTRICT b,
                             T *DSC_RESTRICT c, const int stride_c) {
    // TODO: not aligned!
    vec<T> gamma_0 = load(&c[0 * stride_c]);
    vec<T> gamma_1 = load(&c[1 * stride_c]);
    vec<T> gamma_2 = load(&c[2 * stride_c]);
    vec<T> gamma_3 = load(&c[3 * stride_c]);
    vec<T> gamma_4 = load(&c[4 * stride_c]);
    vec<T> gamma_5 = load(&c[5 * stride_c]);
    vec<T> gamma_6 = load(&c[6 * stride_c]);
    vec<T> gamma_7 = load(&c[7 * stride_c]);
    vec<T> gamma_8 = load(&c[8 * stride_c]);
    vec<T> gamma_9 = load(&c[9 * stride_c]);
    vec<T> gamma_10 = load(&c[10 * stride_c]);
    vec<T> gamma_11 = load(&c[11 * stride_c]);

    vec<T> alpha_pj;

    // Unroll by 4
    const int pb = (k / 4) * 4;
    for (int p = 0; p < pb; p += 4) {
        rank1_avx2(a, b, p + 0);
        rank1_avx2(a, b, p + 1);
        rank1_avx2(a, b, p + 2);
        rank1_avx2(a, b, p + 3);
    }

    // Scalar case
    for (int p = pb; p < k; ++p) {
        rank1_avx2(a, b, p);
    }

    store(&c[0 * stride_c], gamma_0);
    store(&c[1 * stride_c], gamma_1);
    store(&c[2 * stride_c], gamma_2);
    store(&c[3 * stride_c], gamma_3);
    store(&c[4 * stride_c], gamma_4);
    store(&c[5 * stride_c], gamma_5);
    store(&c[6 * stride_c], gamma_6);
    store(&c[7 * stride_c], gamma_7);
    store(&c[8 * stride_c], gamma_8);
    store(&c[9 * stride_c], gamma_9);
    store(&c[10 * stride_c], gamma_10);
    store(&c[11 * stride_c], gamma_11);
}

template<typename T>
consteval usize packed_a_size() {
    return get_gemm_param<T, MC>() * get_gemm_param<T, KC>() * sizeof(T);
}

template<typename T>
consteval usize packed_b_size() {
    return get_gemm_param<T, NC>() * get_gemm_param<T, KC>() * sizeof(T);
}
}

template<typename T, bool trans_b = false>
static DSC_INLINE void dsc_gemm(const int m, const int n, const int k,
                                const T *DSC_RESTRICT a, const int stride_a,
                                T *DSC_RESTRICT packed_a,
                                const T *DSC_RESTRICT b, const int stride_b,
                                T *DSC_RESTRICT packed_b,
                                T *DSC_RESTRICT c, const int stride_c) {
    using namespace internal::gemm;

    const int Nc = get_gemm_param<T, NC>();
    const int Mc = get_gemm_param<T, MC>();
    const int Kc = get_gemm_param<T, KC>();
    const int Mr = get_gemm_param<T, MR>();
    const int Nr = get_gemm_param<T, NR>();

    // 5th loop
    for (int j = 0; j < n; j += Nc) {
        const int jb = DSC_MIN(n - j, Nc);

        // 4th loop
        for (int p = 0; p < k; p += Kc) {
            const int pb = DSC_MIN(k - p, Kc);

            // Pack B
            if constexpr (trans_b) {
                pack_b<T, true>(pb, jb, &b[j * stride_b + p], stride_b, packed_b);
            } else {
                pack_b<T, false>(pb, jb, &b[p * stride_b + j], stride_b, packed_b);
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
                            ukernel_avx2(pb,
                                         &packed_a[ii * pb],
                                         &packed_b[jj * pb],
                                         &c[(i + ii) * stride_c + (j + jj)],
                                         stride_c);
                        } else {
                            // Align to 32 if float and to 64 if double
                            alignas(8 * sizeof(T)) T packed_c[Mr * Nr]{};
                            ukernel_avx2(pb,
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

template<typename T>
static DSC_INLINE void dsc_gevm_transposed(const int n, const int k,
                                           const T *DSC_RESTRICT a,
                                           const T *DSC_RESTRICT b, const int stride_b,
                                           T *DSC_RESTRICT c) {
    // This kernel assumes that the B matrix is transposed in order to access it with stride 1
    for (int j = 0; j < n; ++j) {
        for (int p = 0; p < k; ++p) {
            c[j] += a[p] * b[j * stride_b + p];
        }
    }
}

#undef rank1_avx2