// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <limits>


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
using bf16 = u16;

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

// BF16 on CPU is not a real type, for now we just support its creation but all the operations
// will implicitly convert to F32

// Conversion rules when we have two operands
constexpr static dsc_dtype DSC_DTYPE_CONVERSION_TABLE[DSC_DTYPES][DSC_DTYPES] = {
        {BOOL, I32, F32, F32, F64},
        {I32, I32, F32, F32, F64},
        {F32, F32, F32, F32, F64}, // BF16 is always upcasted to F32
        {F32, F32, F32, F32, F64},
        {F64, F64, F64, F64, F64},
};

constexpr static dsc_dtype DSC_TYPE_AT_LEAST_FLOAT_TABLE[DSC_DTYPES] = {
        F32, // BOOL
        F32, // I32
        F32, // BF16
        F32, // F32
        F64, // F64
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

template<typename T>
consteval T dsc_pi() {
    if constexpr (dsc_is_type<T, f32>()) {
        return 3.14159265358979323846f;
    } else if constexpr (dsc_is_type<T, f64>()){
        return 3.14159265358979323846;
    } else {
        static_assert("T is not supported");
    }
}

template<typename T>
consteval T dsc_zero() {
    if constexpr (dsc_is_type<T, bool>()) {
        return false;
    } else if constexpr (dsc_is_type<T, i32>()) {
        return 0;
    } else if constexpr (dsc_is_type<T, f32>()) {
        return 0.f;
    } else if constexpr (dsc_is_type<T, f64>()) {
        return 0.;
    } else {
        static_assert("T is not supported");
    }
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
