// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "dsc.h"
#include <cmath>

struct cpu_cast_op {
    template<typename Tin, typename Tout>
    DSC_INLINE DSC_STRICTLY_PURE Tout operator()(const Tin in) const {
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

struct cpu_add_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        if constexpr (dsc_is_complex<T>()) {
            return dsc_complex(T, xa.real + xb.real, xa.imag + xb.imag);
        } else {
            return xa + xb;
        }
    }
};

struct cpu_sub_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        if constexpr (dsc_is_complex<T>()) {
            return dsc_complex(T, xa.real - xb.real, xa.imag - xb.imag);
        } else {
            return xa - xb;
        }
    }
};

struct cpu_mul_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        if constexpr (dsc_is_complex<T>()) {
            return dsc_complex(T, (xa.real * xb.real) - (xa.imag * xb.imag),
                               (xa.real * xb.imag) + (xa.imag * xb.real));
        } else {
            return xa * xb;
        }
    }
};

struct cpu_div_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        if constexpr (dsc_is_complex<T>()) {
            return dsc_complex(T, ((xa.real * xb.real) + (xa.imag * xb.imag)) / ((xb.real * xb.real) + (xb.imag * xb.imag)),
                               ((xa.imag * xb.real) - (xa.real * xb.imag)) / ((xb.real * xb.real) + (xb.imag * xb.imag)));
        } else {
            return xa / xb;
        }
    }
};

struct cpu_cos_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T x) const {
        static_assert(dsc_is_complex<T>() || dsc_is_real<T>(), "cpu_cos_op - dtype must be either float or complex");

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

struct cpu_sin_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T x) const {
        static_assert(dsc_is_complex<T>() || dsc_is_real<T>(), "cpu_sin_op - dtype must be either float or complex");

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

struct cpu_sinc_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T x) const {
        static_assert(dsc_is_complex<T>() || dsc_is_real<T>(), "cpu_sinc_op - dtype must be either float or complex");

        static constexpr real<T> pi = dsc_pi<real<T>>();
        static constexpr real<T> zero = dsc_zero<real<T>>();

        if constexpr (dsc_is_real<T>()) {
            const T pi_x = pi * x;
            return (x == zero) ? 1 : cpu_sin_op()(pi_x) / pi_x;
        } else {
            const T pi_x = dsc_complex(T, pi * x.real, pi * x.imag);

            return (x.real == zero && x.imag == zero) ?
                   dsc_complex(T, 1, 0) :
                   cpu_div_op()(cpu_sin_op()(pi_x), pi_x);
        }
    }
};

struct cpu_logn_op {
    template <typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T x) const {
        static_assert(dsc_is_complex<T>() || dsc_is_real<T>(), "cpu_logn_op - dtype must be either float or complex");

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

struct cpu_log2_op {
    template <typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T x) const {
        static_assert(dsc_is_complex<T>() || dsc_is_real<T>(), "cpu_log2_op - dtype must be either float or complex");

        if constexpr (dsc_is_type<T, f32>()) {
            return log2f(x);
        } else if constexpr (dsc_is_type<T, f64>()) {
            return log2(x);
        } else if constexpr (dsc_is_type<T, c32>()) {
            const real<T> fact = 1.4426950408889634f; // 1 / log(2)
            return dsc_complex(T, log2f(sqrtf((x.real * x.real) + (x.imag * x.imag))), fact * atan2f(x.imag, x.real));
        } else if constexpr (dsc_is_type<T, c64>()) {
            const real<T> fact = 1.4426950408889634;
            return dsc_complex(T, log2(sqrt((x.real * x.real) + (x.imag * x.imag))), fact * atan2(x.imag, x.real));
        }
    }
};

struct cpu_log10_op {
    template <typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T x) const {
        static_assert(dsc_is_complex<T>() || dsc_is_real<T>(), "cpu_log10_op - dtype must be either float or complex");

        if constexpr (dsc_is_type<T, f32>()) {
            return log10f(x);
        } else if constexpr (dsc_is_type<T, f64>()) {
            return log10(x);
        } else if constexpr (dsc_is_type<T, c32>()) {
            const real<T> fact = 0.43429448190325176f; // 1 / log(10)
            return dsc_complex(T, log10f(sqrtf((x.real * x.real) + (x.imag * x.imag))), fact * atan2f(x.imag, x.real));
        } else if constexpr (dsc_is_type<T, c64>()) {
            const real<T> fact = 0.43429448190325176;
            return dsc_complex(T, log10(sqrt((x.real * x.real) + (x.imag * x.imag))), fact * atan2(x.imag, x.real));
        }
    }
};

struct cpu_sqrt_op {
    template <typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T x) const {
        static_assert(dsc_is_complex<T>() || dsc_is_real<T>(), "cpu_sqrt_op - dtype must be either float or complex");

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

struct cpu_exp_op {
    template <typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T x) const {
        static_assert(dsc_is_complex<T>() || dsc_is_real<T>(), "cpu_exp_op - dtype must be either float or complex");

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

struct cpu_conj_op {
    template <typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T x) const {
        static_assert(dsc_is_complex<T>(), "cpu_conj_op - dtype must be complex");

        return dsc_complex(T, x.real, -x.imag);
    }
};

struct cpu_real_op {
    template <typename T>
    DSC_INLINE DSC_STRICTLY_PURE real<T> operator()(const T x) const {
        static_assert(dsc_is_complex<T>(), "cpu_real_op - dtype must be complex");

        return x.real;
    }
};

struct cpu_imag_op {
    template <typename T>
    DSC_INLINE DSC_STRICTLY_PURE real<T> operator()(const T x) const {
        static_assert(dsc_is_complex<T>() || dsc_is_real<T>(), "cpu_imag_op - dtype must be either float or complex");

        if constexpr (dsc_is_real<T>()) {
            return dsc_zero<real<T>>();
        } else {
            return x.imag;
        }
    }
};

struct cpu_abs_op {
    template <typename T>
    DSC_INLINE DSC_STRICTLY_PURE real<T> operator()(const T x) const {
        static_assert(dsc_is_complex<T>() || dsc_is_real<T>(), "cpu_abs_op - dtype must be either float or complex");

        if constexpr (dsc_is_real<T>()) {
            return x >= 0 ? x : -x;
        } else if constexpr (dsc_is_type<T, c32>()){
            return sqrtf((x.real * x.real) + (x.imag * x.imag));
        } else {
            return sqrt((x.real * x.real) + (x.imag * x.imag));
        }
    }
};

struct cpu_atan2_op {
    template <typename T>
    DSC_INLINE DSC_STRICTLY_PURE real<T> operator()(const T x) const {
        static_assert(dsc_is_complex<T>() || dsc_is_real<T>(), "cpu_atan2_op - dtype must be either float or complex");

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

struct cpu_pow_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T base, const T exp) const {
        if constexpr (dsc_is_type<f32, T>()) {
            return powf(base, exp);
        } else if constexpr (dsc_is_type<f64, T>()) {
            return pow(base, exp);
        } else {
            return cpu_exp_op()(cpu_mul_op()(exp, cpu_logn_op()(base)));
        }
    }
};

struct cpu_max_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        if constexpr (dsc_is_real<T>()) {
            return DSC_MAX(xa, xb);
        } else {
            // This is NumPy behaviour
            return xa.real > xb.real ? xa : xb;
        }
    }
};

struct cpu_min_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        if constexpr (dsc_is_real<T>()) {
            return DSC_MIN(xa, xb);
        } else {
            return xa.real > xb.real ? xb : xa;
        }
    }
};

struct cpu_i0_op {
    template <typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T x) const {
        static_assert(dsc_is_real<T>(), "T must be real");

        // Taken from Numerical Recipes
        if constexpr (dsc_is_type<T, f32>()) {
            f32 ax, y, res;
            if ((ax = fabsf(x)) < 3.75f) {
                y = x / 3.75f;
                y *= y;
                res = 1.f + y * (3.5156229f + y *
                                (3.0899424f + y *
                                (1.2067492f + y *
                                (0.2659732f + y *
                                (0.360768e-1f + y *
                                (0.45813e-2f)))))
                );
            } else {
                y = 3.75f / ax;
                res = (expf(ax) / sqrtf(ax)) *
                      (0.39894228f + y *
                      (0.1328592e-1f + y *
                      (0.225319e-2f + y *
                      (-0.157565e-2f + y *
                      (0.916281e-2f + y *
                      (-0.2057706e-1f + y *
                      (0.2635537e-1f + y *
                      (-0.1647633e-1f + y *
                      (0.392377e-2f))))))))
                );
            }

            return res;
        } else {
            f64 ax, y, res;
            if ((ax = fabs(x)) < 3.75) {
                y = x / 3.75;
                y *= y;
                res = 1. + y * (3.5156229 + y *
                               (3.0899424 + y *
                               (1.2067492 + y *
                               (0.2659732 + y *
                               (0.360768e-1 + y *
                               (0.45813e-2)))))
                );
            } else {
                y = 3.75 / ax;
                res = (exp(ax) / sqrt(ax)) *
                      (0.39894228 + y *
                      (0.1328592e-1 + y *
                      (0.225319e-2 + y *
                      (-0.157565e-2 + y *
                      (0.916281e-2 + y *
                      (-0.2057706e-1 + y *
                      (0.2635537e-1 + y *
                      (-0.1647633e-1 + y *
                      (0.392377e-2))))))))
                );
            }

            return res;
        }
    }
};

template<typename T>
struct cpu_clip_op {
    cpu_clip_op(const T x_min, const T x_max) : x_min_(x_min), x_max_(x_max) {}

    DSC_INLINE DSC_PURE T operator()(const T x) const {
        return cpu_min_op()(
                cpu_max_op()(x, x_min_),
                x_max_);
    }

private:
    const T x_min_, x_max_;
};