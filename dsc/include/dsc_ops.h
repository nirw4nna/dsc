// Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include <cmath>
#include "dsc.h"

struct cast_op {
    template<typename Tin, typename Tout>
    DSC_INLINE DSC_STRICTLY_PURE Tout operator()(const Tin in) noexcept {
        if constexpr (dsc_is_complex<Tout>()) {
            if constexpr (dsc_is_real<Tin>()) {
                if constexpr (dsc_is_type<Tout, c32>()) {
                    return dsc_complex(Tout, (f32) in, 0);
                }
                if constexpr (dsc_is_type<Tout, c64>()) {
                    return dsc_complex(Tout, (f64) in, 0);
                }
            } else {
                if constexpr (dsc_is_type<Tout, c32>()) {
                    return dsc_complex(Tout, (f32) in.real, (f32) in.imag);
                }
                if constexpr (dsc_is_type<Tout, c64>()) {
                    return dsc_complex(Tout, (f64) in.real, (f64) in.imag);
                }
            }
        } else {
            if constexpr (dsc_is_real<Tin>()) {
                return (Tout) in;
            } else {
                if constexpr (dsc_is_type<Tout, f32>()) {
                    return (f32) in.real;
                }
                if constexpr (dsc_is_type<Tout, f64>()) {
                    return (f64) in.real;
                }
            }
        }
    }
};

struct add_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const noexcept {
        if constexpr (dsc_is_complex<T>()) {
            return dsc_complex(T, xa.real + xb.real, xa.imag + xb.imag);
        } else {
            return xa + xb;
        }
    }
};

struct sub_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const noexcept {
        if constexpr (dsc_is_complex<T>()) {
            return dsc_complex(T, xa.real - xb.real, xa.imag - xb.imag);
        } else {
            return xa - xb;
        }
    }
};

struct mul_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const noexcept {
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
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const noexcept {
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
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T x) const noexcept {
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
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T x) const noexcept {
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

struct sinc_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T x) const noexcept {
        static_assert(dsc_is_complex<T>() || dsc_is_real<T>(), "sinc_op - dtype must be either float or complex");

        static constexpr real<T> pi = dsc_pi<real<T>>();
        static constexpr real<T> zero = dsc_zero<real<T>>();

        if constexpr (dsc_is_real<T>()) {
            const T pi_x = pi * x;
            return (x == zero) ? 1 : sin_op()(pi_x) / pi_x;
        } else {
            const T pi_x = dsc_complex(T, pi * x.real, pi * x.imag);

            return (x.real == zero && x.imag == zero) ?
                   dsc_complex(T, 1, 0) :
                   div_op()(sin_op()(pi_x), pi_x);
        }
    }
};

struct logn_op {
    template <typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T x) const noexcept {
        static_assert(dsc_is_complex<T>() || dsc_is_real<T>(), "logn_op - dtype must be either float or complex");

        if constexpr (dsc_is_type<T, f32>()) {
            return logf(x);
        } else if constexpr (dsc_is_type<T, f64>()) {
            return log(x);
        } else if constexpr (dsc_is_type<T, c32>()) {
            return dsc_complex(T, logf(sqrtf((x.real * x.real) + (x.imag * x.imag))), atan2f(x.imag, x.real));
        } else if constexpr (dsc_is_type<T, c64>()) {
            return dsc_complex(T, log(sqrt((x.real * x.real) + (x.imag * x.imag))), atan2(x.imag, x.real));
        }
    }
};

struct log2_op {
    template <typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T x) const noexcept {
        static_assert(dsc_is_complex<T>() || dsc_is_real<T>(), "log2_op - dtype must be either float or complex");

        if constexpr (dsc_is_type<T, f32>()) {
            return log2f(x);
        } else if constexpr (dsc_is_type<T, f64>()) {
            return log2(x);
        } else if constexpr (dsc_is_type<T, c32>()) {
            static constexpr real<T> fact = 1 / logf(2);
            return dsc_complex(T, log2f(sqrtf((x.real * x.real) + (x.imag * x.imag))), fact * atan2f(x.imag, x.real));
        } else if constexpr (dsc_is_type<T, c64>()) {
            static constexpr real<T> fact = 1 / log(2);
            return dsc_complex(T, log2(sqrt((x.real * x.real) + (x.imag * x.imag))), fact * atan2(x.imag, x.real));
        }
    }
};

struct log10_op {
    template <typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T x) const noexcept {
        static_assert(dsc_is_complex<T>() || dsc_is_real<T>(), "log10_op - dtype must be either float or complex");

        if constexpr (dsc_is_type<T, f32>()) {
            return log10f(x);
        } else if constexpr (dsc_is_type<T, f64>()) {
            return log10(x);
        } else if constexpr (dsc_is_type<T, c32>()) {
            static constexpr real<T> fact = 1 / logf(10);
            return dsc_complex(T, log10f(sqrtf((x.real * x.real) + (x.imag * x.imag))), fact * atan2f(x.imag, x.real));
        } else if constexpr (dsc_is_type<T, c64>()) {
            static constexpr real<T> fact = 1 / log(10);
            return dsc_complex(T, log10(sqrt((x.real * x.real) + (x.imag * x.imag))), fact * atan2(x.imag, x.real));
        }
    }
};

struct sqrt_op {
    template <typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T x) const noexcept {
        static_assert(dsc_is_complex<T>() || dsc_is_real<T>(), "sqrt_op - dtype must be either float or complex");

        if constexpr (dsc_is_type<T, f32>()) {
            return sqrtf(x);
        } else if constexpr (dsc_is_type<T, f64>()) {
            return sqrt(x);
        } else if constexpr (dsc_is_type<T, c32>()) {
            const real<T> abs = sqrtf((x.real * x.real) + (x.imag * x.imag));
            const real<T> sign = x.imag >= 0 ? 1 : -1;
            return dsc_complex(T, sqrtf(0.5f * (abs + x.real)), sign * sqrtf(0.5f * (abs - x.real)));
        } else if constexpr (dsc_is_type<T, c64>()) {
            const real<T> abs = sqrt((x.real * x.real) + (x.imag * x.imag));
            const real<T> sign = x.imag >= 0 ? 1 : -1;
            return dsc_complex(T, sqrt(0.5 * (abs + x.real)), sign * sqrt(0.5 * (abs - x.real)));
        }
    }
};

struct exp_op {
    template <typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T x) const noexcept {
        static_assert(dsc_is_complex<T>() || dsc_is_real<T>(), "exp_op - dtype must be either float or complex");

        if constexpr (dsc_is_type<T, f32>()) {
            return expf(x);
        } else if constexpr (dsc_is_type<T, f64>()) {
            return exp(x);
        } else if constexpr (dsc_is_type<T, c32>()) {
            const real<T> fact = expf(x.real);
            return dsc_complex(T, fact * cosf(x.imag), fact * sinf(x.imag));
        } else if constexpr (dsc_is_type<T, c64>()) {
            const real<T> fact = exp(x.real);
            return dsc_complex(T, fact * cos(x.imag), fact * sin(x.imag));
        }
    }
};

struct conj_op {
    template <typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T x) const noexcept {
        static_assert(dsc_is_complex<T>(), "conj_op - dtype must be complex");

        return dsc_complex(T, x.real, -x.imag);
    }
};

struct real_op {
    template <typename T>
    DSC_INLINE DSC_STRICTLY_PURE real<T> operator()(const T x) const noexcept {
        static_assert(dsc_is_complex<T>(), "real_op - dtype must be complex");

        return x.real;
    }
};

struct imag_op {
    template <typename T>
    DSC_INLINE DSC_STRICTLY_PURE real<T> operator()(const T x) const noexcept {
        static_assert(dsc_is_complex<T>() || dsc_is_real<T>(), "imag_op - dtype must be either float or complex");

        if constexpr (dsc_is_real<T>()) {
            return dsc_zero<real<T>>();
        } else {
            return x.imag;
        }
    }
};

struct abs_op {
    template <typename T>
    DSC_INLINE DSC_STRICTLY_PURE real<T> operator()(const T x) const noexcept {
        static_assert(dsc_is_complex<T>() || dsc_is_real<T>(), "abs_op - dtype must be either float or complex");

        if constexpr (dsc_is_real<T>()) {
            return x >= 0 ? x : -x;
        } else if constexpr (dsc_is_type<T, c32>()){
            return sqrtf((x.real * x.real) + (x.imag * x.imag));
        } else {
            return sqrt((x.real * x.real) + (x.imag * x.imag));
        }
    }
};

struct atan2_op {
    template <typename T>
    DSC_INLINE DSC_STRICTLY_PURE real<T> operator()(const T x) const noexcept {
        static_assert(dsc_is_complex<T>() || dsc_is_real<T>(), "atan2_op - dtype must be either float or complex");

        if constexpr (dsc_is_type<T, f32>()) {
            return atan2f(0.f, x);
        } else if constexpr (dsc_is_type<T, f64>()) {
            return atan2(0., x);
        } else if constexpr (dsc_is_type<T, c32>()) {
            return atan2f(x.imag, x.real);
        } else {
            return atan2(x.imag, x.real);
        }
    }
};

struct pow_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T base, const T exp) const noexcept {
        if constexpr (dsc_is_type<f32, T>()) {
            return powf(base, exp);
        } else if constexpr (dsc_is_type<f64, T>()) {
            return pow(base, exp);
        } else {
            return exp_op()(mul_op()(exp, logn_op()(base)));
        }
    }
};