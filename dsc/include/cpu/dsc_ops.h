// Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
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
        if constexpr (dsc_is_type<Tin, bf16>()) {
            // Naive way of converting between BF16 and F32, if this has to be applied to a sequence of
            // elements it can be vectorized quite easily.
            union {
                f32 f;
                u32 i;
            } u;
            u.i = (u32) in << 16;

            return (Tout) u.f;
        } else {
            return (Tout) in;
        }
    }
};

struct cpu_add_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        if constexpr (dsc_is_type<T, bool>()) {
            return xa || xb;
        } else {
            return xa + xb;
        }
    }
};

struct cpu_sub_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        if constexpr (dsc_is_type<T, bool>()) {
            return xa ^ xb;
        } else {
            return xa - xb;
        }
    }
};

struct cpu_mul_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        if constexpr (dsc_is_type<T, bool>()) {
            return xa && xb;
        } else {
            return xa * xb;
        }
    }
};

struct cpu_div_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        return xa / xb;
    }
};

struct cpu_pow_op {
    DSC_INLINE DSC_STRICTLY_PURE i32 operator()(const i32 base, const i32 exp) const {
        i32 acc = 1;
        for (int i = 0; i < exp; ++i) acc *= base;
        return acc;
    }

    DSC_INLINE DSC_STRICTLY_PURE f32 operator()(const f32 base, const f32 exp) const {
        return powf(base, exp);
    }

    DSC_INLINE DSC_STRICTLY_PURE f64 operator()(const f64 base, const f64 exp) const {
        return pow(base, exp);
    }
};

struct cpu_cos_op {
    DSC_INLINE DSC_STRICTLY_PURE f32 operator()(const f32 x) const {
        return cosf(x);
    }

    DSC_INLINE DSC_STRICTLY_PURE f64 operator()(const f64 x) const {
        return cos(x);
    }
};

struct cpu_sin_op {
    DSC_INLINE DSC_STRICTLY_PURE f32 operator()(const f32 x) const {
        return sinf(x);
    }

    DSC_INLINE DSC_STRICTLY_PURE f64 operator()(const f64 x) const {
        return sin(x);
    }
};

struct cpu_tanh_op {
    DSC_INLINE DSC_STRICTLY_PURE f32 operator()(const f32 x) const {
        return tanhf(x);
    }

    DSC_INLINE DSC_STRICTLY_PURE f64 operator()(const f64 x) const {
        return tanh(x);
    }
};

struct cpu_sqrt_op {
    DSC_INLINE DSC_STRICTLY_PURE f32 operator()(const f32 x) const {
        return sqrtf(x);
    }

    DSC_INLINE DSC_STRICTLY_PURE f64 operator()(const f64 x) const {
        return sqrt(x);
    }
};

struct cpu_exp_op {
    DSC_INLINE DSC_STRICTLY_PURE f32 operator()(const f32 x) const {
        return expf(x);
    }

    DSC_INLINE DSC_STRICTLY_PURE f64 operator()(const f64 x) const {
        return exp(x);
    }
};

struct cpu_max_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        return DSC_MAX(xa, xb);
    }
};

struct cpu_min_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        return DSC_MIN(xa, xb);
    }
};

struct cpu_eq_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE bool operator()(const T xa, const T xb) const {
        return xa == xb;
    }
};

struct cpu_ne_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE bool operator()(const T xa, const T xb) const {
        return !cpu_eq_op()(xa, xb);
    }
};

struct cpu_lt_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE bool operator()(const T xa, const T xb) const {
        return xa < xb;
    }
};
struct cpu_le_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE bool operator()(const T xa, const T xb) const {
        return xa <= xb;
    }
};

struct cpu_gt_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE bool operator()(const T xa, const T xb) const {
        return xa > xb;
    }
};

struct cpu_ge_op {
    template<typename T>
    DSC_INLINE DSC_STRICTLY_PURE bool operator()(const T xa, const T xb) const {
        return xa >= xb;
    }
};

template<typename Op>
consteval bool is_comparison_op() {
    return dsc_is_type<Op, cpu_eq_op>() ||
           dsc_is_type<Op, cpu_ne_op>() ||
           dsc_is_type<Op, cpu_lt_op>() ||
           dsc_is_type<Op, cpu_le_op>() ||
           dsc_is_type<Op, cpu_gt_op>() ||
           dsc_is_type<Op, cpu_ge_op>();
}

template<typename Op>
consteval bool is_bool_arith_op() {
    return dsc_is_type<Op, cpu_add_op>() ||
           dsc_is_type<Op, cpu_sub_op>() ||
           dsc_is_type<Op, cpu_mul_op>();
}