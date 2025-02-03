// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "dsc.h"
#include <immintrin.h>

#define DSC_SIMD_WIDTH      ((int) 256)
#define dsc_prefetch(ADDR)  _mm_prefetch((ADDR), _MM_HINT_T0)

using f32x8 = __m256;
using f64x4 = __m256d;

namespace {
template<typename T>
struct vec_;

template<>
struct vec_<f32> {
    using type = f32x8;
};

template<>
struct vec_<f64> {
    using type = f64x4;
};
}

template <typename T>
using vec = typename vec_<T>::type;

template<typename T>
static DSC_INLINE vec<T> load(const T *DSC_RESTRICT addr) {
    static_assert(dsc_is_real<T>(), "load - T must be real");

    if constexpr (dsc_is_type<T, f32>()) {
        return _mm256_loadu_ps(addr);
    } else {
        return _mm256_loadu_pd(addr);
    }
}

template<typename T>
static DSC_INLINE void store(T *DSC_RESTRICT addr, const vec<T> reg) {
    static_assert(dsc_is_real<T>(), "store - T must be real");

    if constexpr (dsc_is_type<T, f32>()) {
        _mm256_storeu_ps(addr, reg);
    } else {
        _mm256_storeu_pd(addr, reg);
    }
}

template<typename T>
static DSC_INLINE vec<T> broadcast(const T *DSC_RESTRICT addr) {
    static_assert(dsc_is_real<T>(), "broadcast - T must be real");

    if constexpr (dsc_is_type<T, f32>()) {
        return _mm256_broadcast_ss(addr);
    } else {
        return _mm256_broadcast_sd(addr);
    }
}

template<typename V>
static DSC_INLINE V fmadd(const V a, const V b, const V c) {
    static_assert(dsc_is_type<V, f32x8>() || dsc_is_type<V, f64x4>(), "fmadd - V must be real");

    if constexpr (dsc_is_type<V, f32x8>()) {
        return _mm256_fmadd_ps(a, b, c);
    } else {
        return _mm256_fmadd_pd(a, b, c);
    }
}