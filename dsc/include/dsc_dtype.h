#pragma once

#include <complex>
#include <cstdint>
#include <cstddef>
#include <type_traits>


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

namespace {
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

using c32 = complex_<f32>;
using c64 = complex_<f64>;

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
        {dsc_dtype::F32, dsc_dtype::F64, dsc_dtype::C32, dsc_dtype::C64},
        {dsc_dtype::F64, dsc_dtype::F64, dsc_dtype::C32, dsc_dtype::C64},
        {dsc_dtype::C32, dsc_dtype::C32, dsc_dtype::C32, dsc_dtype::C64},
        {dsc_dtype::C64, dsc_dtype::C64, dsc_dtype::C64, dsc_dtype::C64},
};

template<typename Ta, typename Tb>
static consteval bool dsc_is_type() noexcept {
    return std::is_same_v<Ta, Tb>;
}

template<typename T>
static consteval bool dsc_is_complex() noexcept {
    return dsc_is_type<T, c32>() || dsc_is_type<T, c64>();
}

template<typename T>
static consteval bool dsc_is_real() noexcept {
    return dsc_is_type<T, f32>() || dsc_is_type<T, f64>();
}

//template<typename T>
//static consteval dsc_dtype dsc_complex_base_type() noexcept {
//    static_assert(dsc_is_complex<T>());
//
//    if constexpr (dsc_is_type<T, c32>()) {
//        return dsc_dtype::F32;
//    } else {
//        return dsc_dtype::F64;
//    }
//}

// Conversion utility
template<typename T>
struct dsc_type_mapping;

template<>
struct dsc_type_mapping<f32> {
    static constexpr dsc_dtype value = dsc_dtype::F32;
};

template<>
struct dsc_type_mapping<f64> {
    static constexpr dsc_dtype value = dsc_dtype::F64;
};

template<>
struct dsc_type_mapping<c32> {
    static constexpr dsc_dtype value = dsc_dtype::C32;
};

template<>
struct dsc_type_mapping<c64> {
    static constexpr dsc_dtype value = dsc_dtype::C64;
};
