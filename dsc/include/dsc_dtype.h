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


#define DSC_DTYPES       ((int) 4)
#define DSC_DEFAULT_TYPE (dsc_dtype::F32)

#define dsc_complex(type, real, imag) (type{.d = {(real), (imag)}})


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
using ulonglong = unsigned long long int;

namespace internal {
template<typename T>
struct complex_ {
    union {
        T d[2];
        struct {
            T real, imag;
        };
    };
};
}

using c32 = internal::complex_<f32>;
using c64 = internal::complex_<f64>;

enum dsc_dtype : u8 {
    F32,
    F64,
    C32,
    C64,
};

constexpr static usize DSC_DTYPE_SIZE[DSC_DTYPES] = {
        sizeof(f32),
        sizeof(f64),
        sizeof(c32),
        sizeof(c64)
};

constexpr static const char *DSC_DTYPE_NAMES[DSC_DTYPES] = {
        "f32",
        "f64",
        "c32",
        "c64",
};

// Conversion rules when we have two operands
constexpr static dsc_dtype DSC_DTYPE_CONVERSION_TABLE[DSC_DTYPES][DSC_DTYPES] = {
        {F32, F64, C32, C64},
        {F64, F64, C32, C64},
        {C32, C32, C32, C64},
        {C64, C64, C64, C64},
};

constexpr static dsc_dtype DSC_DTYPE_TO_REAL[DSC_DTYPES] = {
        F32,// F32
        F64,// F64
        F32,// C32
        F64 // C64
};

// Conversion utility
template<typename T>
struct dsc_type_mapping;

template<>
struct dsc_type_mapping<f32> {
    static constexpr dsc_dtype value = F32;
};

template<>
struct dsc_type_mapping<f64> {
    static constexpr dsc_dtype value = F64;
};

template<>
struct dsc_type_mapping<c32> {
    static constexpr dsc_dtype value = C32;
};

template<>
struct dsc_type_mapping<c64> {
    static constexpr dsc_dtype value = C64;
};

namespace {
    template<typename T>
    struct real_;

    template<>
    struct real_<c32> {
        using type = f32;
    };

    template<>
    struct real_<c64> {
        using type = f64;
    };

    template<>
    struct real_<f32> {
        using type = f32;
    };

    template<>
    struct real_<f64> {
        using type = f64;
    };
}

template <typename T>
using real = typename real_<T>::type;

template<typename Ta, typename Tb>
static consteval bool dsc_is_type() {
    return std::is_same_v<Ta, Tb>;
}

template<typename T>
static consteval bool dsc_is_complex() {
    return dsc_is_type<T, c32>() || dsc_is_type<T, c64>();
}

template<typename T>
static consteval bool dsc_is_real() {
    return dsc_is_type<T, f32>() || dsc_is_type<T, f64>();
}

template<typename T>
static consteval T dsc_pi() {
    if constexpr (dsc_is_type<T, f32>()) {
        return 3.14159265358979323846f;
    } else {
        return 3.14159265358979323846;
    }
}

template<typename T>
static consteval T dsc_zero() {
    if constexpr (dsc_is_type<T, f32>()) {
        return 0.f;
    } else if constexpr (dsc_is_type<T, f64>()) {
        return 0;
    } else if constexpr (dsc_is_type<T, c32>()) {
        return dsc_complex(c32, 0.f, 0.f);
    } else if constexpr (dsc_is_type<T, c64>()) {
        return dsc_complex(c64, 0., 0.);
    } else {
        static_assert("T is not supported");
    }
}

template<typename T, bool positive = true>
static consteval T dsc_inf() {
    constexpr real<T> sign = positive ? 1 : -1;
    if constexpr (dsc_is_type<T, f32>()) {
        return sign * std::numeric_limits<f32>::infinity();
    } else if constexpr (dsc_is_type<T, f64>()) {
        return sign * std::numeric_limits<f64>::infinity();
    } else if constexpr (dsc_is_type<T, c32>()) {
        return dsc_complex(c32, sign * std::numeric_limits<f32>::infinity(), sign * std::numeric_limits<f32>::infinity());
    } else if constexpr (dsc_is_type<T, c64>()) {
        return dsc_complex(c64, sign * std::numeric_limits<f64>::infinity(), sign * std::numeric_limits<f64>::infinity());
    } else {
        static_assert("T is not supported");
    }
}
