#pragma once

#include <cmath>
#include "dsc.h"


struct add_op {
    template<typename T>
    DSC_INLINE T operator()(const T xa, const T xb) const noexcept {
        if constexpr (dsc_is_complex<T>()) {
            return dsc_complex(T, xa.real + xb.real, xa.imag + xb.imag);
        } else {
            return xa + xb;
        }
    }
};

struct sub_op {
    template<typename T>
    DSC_INLINE T operator()(const T xa, const T xb) const noexcept {
        if constexpr (dsc_is_complex<T>()) {
            return dsc_complex(T, xa.real - xb.real, xa.imag - xb.imag);
        } else {
            return xa - xb;
        }
    }
};

struct mul_op {
    template<typename T>
    DSC_INLINE T operator()(const T xa, const T xb) const noexcept {
        if constexpr (dsc_is_complex<T>()) {
            return dsc_complex(T, (xa.real * xb.real) - (xa.imag * xb.imag),
                               (xa.real * xb.imag) + (xa.imag * xb.real));
        } else {
            return xa * xb;
        }
    }
};

struct div_op {
    template<typename T>
    DSC_INLINE T operator()(const T xa, const T xb) const noexcept {
        if constexpr (dsc_is_complex<T>()) {
            return dsc_complex(T, ((xa.real * xb.real) + (xa.imag * xb.imag)) / ((xb.real * xb.real) + (xb.imag * xb.imag)),
                               ((xa.imag * xb.real) - (xa.real * xb.imag)) / ((xb.real * xb.real) + (xb.imag * xb.imag)));
        } else {
            return xa / xb;
        }
    }
};

struct cos_op {
    template<typename T>
    DSC_INLINE T operator()(const T x) const noexcept {
        static_assert(dsc_is_complex<T>() || dsc_is_real<T>(), "cos_op - dtype must be either float or complex");

        if constexpr (dsc_is_type<T, f32>()) {
            return cosf(x);
        } else if constexpr (dsc_is_type<T, f64>()) {
            return cos(x);
        } else if constexpr (dsc_is_type<T, c32>()) {
            return dsc_complex(T, cosf(x.real) * coshf(x.imag), -sinf(x.real) * sinhf(x.imag));
        } else if constexpr (dsc_is_type<T, c64>()) {
            return dsc_complex(T, cos(x.real) * cosh(x.imag), -sin(x.real) * sinh(x.imag));
        }
    }
};

struct sin_op {
    template<typename T>
    DSC_INLINE T operator()(const T x) const noexcept {
        static_assert(dsc_is_complex<T>() || dsc_is_real<T>(), "sin_op - dtype must be either float or complex");

        if constexpr (dsc_is_type<T, f32>()) {
            return sinf(x);
        } else if constexpr (dsc_is_type<T, f64>()) {
            return sin(x);
        } else if constexpr (dsc_is_type<T, c32>()) {
            return dsc_complex(T, sinf(x.real) * coshf(x.imag), cosf(x.real) * sinhf(x.imag));
        } else if constexpr (dsc_is_type<T, c64>()) {
            return dsc_complex(T, sin(x.real) * cosh(x.imag), cos(x.real) * sinh(x.imag));
        }
    }
};
