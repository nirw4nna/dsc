// Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(__HIPCC__) && defined(DSC_BF16)
#   include <hip/hip_bf16.h>
#endif

#define DSC_DTYPES          ((int) 5)
#define DSC_DEFAULT_TYPE    F32


using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using size = ptrdiff_t;
using usize = size_t;
using byte = char;
using f32 = float;
using f64 = double;
#if defined(__HIPCC__) && defined(DSC_BF16)
    using bf16 = __hip_bfloat16;
#else
    using bf16 = u16;
#endif

enum dsc_dtype : u8 {
    BOOL,
    I32,
    BF16,
    F32,
    F64,
};

constexpr static usize DSC_DTYPE_SIZE[DSC_DTYPES] = {
        sizeof(bool),
        sizeof(i32),
        sizeof(bf16),
        sizeof(f32),
        sizeof(f64),
};

constexpr static const char *DSC_DTYPE_NAMES[DSC_DTYPES] = {
        "bool",
        "i32",
        "bf16",
        "f32",
        "f64",
};


// Conversion utility
template<typename T>
struct dsc_type_mapping;

template<>
struct dsc_type_mapping<bool> {
    static constexpr dsc_dtype value = BOOL;
};

template<>
struct dsc_type_mapping<i32> {
    static constexpr dsc_dtype value = I32;
};

template<>
struct dsc_type_mapping<bf16> {
    static constexpr dsc_dtype value = BF16;
};

template<>
struct dsc_type_mapping<f32> {
    static constexpr dsc_dtype value = F32;
};

template<>
struct dsc_type_mapping<f64> {
    static constexpr dsc_dtype value = F64;
};

template<typename Ta, typename Tb>
consteval bool dsc_is_type() {
    return std::is_same_v<Ta, Tb>;
}

template<typename T>
consteval bool dsc_is_real() {
    return dsc_is_type<T, bf16>() || dsc_is_type<T, f32>() || dsc_is_type<T, f64>();
}

template<typename T, bool positive = true>
consteval T dsc_inf() {
    constexpr T sign = positive ? 1 : -1;

    if constexpr (dsc_is_type<T, f32>()) {
        return sign * std::numeric_limits<f32>::infinity();
    } else if constexpr (dsc_is_type<T, f64>()) {
        return sign * std::numeric_limits<f64>::infinity();
    } else {
        static_assert("T is not supported");
    }
}
