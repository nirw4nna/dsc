// Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "gpu/dsc_gpu.h"

#define atomic_cas_f32(PTR, VAL)                                     \
    do {                                                             \
        uint *addr = (uint *) (PTR);                                 \
        uint old = *addr, assumed;                                   \
        do {                                                         \
            assumed = old;                                           \
            const f32 assumed_val = __int_as_float(assumed);         \
            const f32 new_val = VAL;                                 \
            old = atomicCAS(addr, assumed, __float_as_int(new_val)); \
        } while (old != assumed);                                    \
    } while (0)

#define atomic_cas_f64(PTR, VAL)                                           \
    do {                                                                   \
        unsigned long long *addr = (unsigned long long *) (PTR);           \
        unsigned long long old = *addr, assumed;                           \
        do {                                                               \
            assumed = old;                                                 \
            const f64 assumed_val = __longlong_as_double(assumed);         \
            const f64 new_val = VAL;                                       \
            old = atomicCAS(addr, assumed, __double_as_longlong(new_val)); \
        } while (old != assumed);                                          \
    } while (0)


struct gpu_cast_op {
    template<typename Tin, typename Tout>
    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE Tout operator()(const Tin in) const {
#if defined(DSC_BF16)
        return (Tout) in;
#else
        if constexpr (dsc_is_type<Tin, bf16>()) {
            // If BF16 is not supported use the same logic as the CPU
            union {
                f32 f;
                u32 i;
            } u;
            u.i = (u32) in << 16;
            return (Tout) u.f;
        } else {
            return (Tout) in;
        }
#endif
    }
};

struct gpu_add_op {
    template<typename T>
    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        if constexpr (dsc_is_type<T, bool>()) {
            return xa || xb;
        } else {
            return xa + xb;
        }
    }
};

struct gpu_atomic_add_op {
    template<typename T>
    DSC_GPU_FUNC DSC_INLINE void operator()(T *x, const T val) const {
        if constexpr (dsc_is_type<T, bool>()) {
            atomicOr(x, val);
        } else {
            atomicAdd(x, val);
        }
    }
};

struct gpu_sub_op {
    template<typename T>
    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        if constexpr (dsc_is_type<T, bool>()) {
            return xa ^ xb;
        } else {
            return xa - xb;
        }
    }
};

struct gpu_mul_op {
    template<typename T>
    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        if constexpr (dsc_is_type<T, bool>()) {
            return xa && xb;
        } else {
            return xa * xb;
        }
    }
};

struct gpu_div_op {
    template<typename T>
    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        return xa / xb;
    }
};

struct gpu_pow_op {
    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE i32 operator()(const i32 base, const i32 exp) const {
        i32 acc = 1;
        for (int i = 0; i < exp; ++i) acc *= base;
        return acc;
    }

    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE bf16 operator()(const bf16 base, const bf16 exp) const {
        return gpu_pow_op()((f32) base, (f32) exp);
    }

    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE f32 operator()(const f32 base, const f32 exp) const {
        return powf(base, exp);
    }

    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE f64 operator()(const f64 base, const f64 exp) const {
        return pow(base, exp);
    }
};

struct gpu_cos_op {
    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE bf16 operator()(const bf16 x) const {
        return gpu_cos_op()((f32) x);
    }

    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE f32 operator()(const f32 x) const {
        return cosf(x);
    }

    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE f64 operator()(const f64 x) const {
        return cos(x);
    }
};

struct gpu_sin_op {
    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE bf16 operator()(const bf16 x) const {
        return gpu_sin_op()((f32) x);
    }

    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE f32 operator()(const f32 x) const {
        return sinf(x);
    }

    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE f64 operator()(const f64 x) const {
        return sin(x);
    }
};

struct gpu_tanh_op {
    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE bf16 operator()(const bf16 x) const {
        return gpu_tanh_op()((f32) x);
    }

    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE f32 operator()(const f32 x) const {
        return tanhf(x);
    }

    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE f64 operator()(const f64 x) const {
        return tanh(x);
    }
};

struct gpu_sqrt_op {
    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE bf16 operator()(const bf16 x) const {
        return gpu_sqrt_op()((f32) x);
    }

    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE f32 operator()(const f32 x) const {
        return sqrtf(x);
    }

    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE f64 operator()(const f64 x) const {
        return sqrt(x);
    }
};

struct gpu_exp_op {
    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE bf16 operator()(const bf16 x) const {
        return gpu_exp_op()((f32) x);
    }

    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE f32 operator()(const f32 x) const {
        return expf(x);
    }

    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE f64 operator()(const f64 x) const {
        return exp(x);
    }
};

struct gpu_max_op {
    template<typename T>
    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        return DSC_MAX(xa, xb);
    }
};

struct gpu_atomic_max_op {
    DSC_GPU_FUNC DSC_INLINE void operator()(f32 *x, const f32 val) const {
        atomic_cas_f32(x, DSC_MAX(val, assumed_val));
    }

    DSC_GPU_FUNC DSC_INLINE void operator()(f64 *x, const f64 val) const {
        atomic_cas_f64(x, DSC_MAX(val, assumed_val));
    }
};

struct gpu_min_op {
    template<typename T>
    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        return DSC_MIN(xa, xb);
    }
};

struct gpu_atomic_min_op {
    DSC_GPU_FUNC DSC_INLINE void operator()(f32 *x, const f32 val) const {
        atomic_cas_f32(x, DSC_MIN(val, assumed_val));
    }

    DSC_GPU_FUNC DSC_INLINE void operator()(f64 *x, const f64 val) const {
        atomic_cas_f64(x, DSC_MIN(val, assumed_val));
    }
};

struct gpu_eq_op {
    template<typename T>
    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE bool operator()(const T xa, const T xb) const {
        return xa == xb;
    }
};

struct gpu_ne_op {
    template<typename T>
    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE bool operator()(const T xa, const T xb) const {
        return !gpu_eq_op()(xa, xb);
    }
};

struct gpu_lt_op {
    template<typename T>
    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE bool operator()(const T xa, const T xb) const {
        return xa < xb;
    }
};
struct gpu_le_op {
    template<typename T>
    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE bool operator()(const T xa, const T xb) const {
        return xa <= xb;
    }
};

struct gpu_gt_op {
    template<typename T>
    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE bool operator()(const T xa, const T xb) const {
        return xa > xb;
    }
};

struct gpu_ge_op {
    template<typename T>
    DSC_GPU_FUNC DSC_INLINE DSC_STRICTLY_PURE bool operator()(const T xa, const T xb) const {
        return xa >= xb;
    }
};


template<typename Op>
consteval bool is_comparison_op() {
    return dsc_is_type<Op, gpu_eq_op>() ||
           dsc_is_type<Op, gpu_ne_op>() ||
           dsc_is_type<Op, gpu_lt_op>() ||
           dsc_is_type<Op, gpu_le_op>() ||
           dsc_is_type<Op, gpu_gt_op>() ||
           dsc_is_type<Op, gpu_ge_op>();
}

template<typename Op>
consteval bool is_bool_arith_op() {
    return dsc_is_type<Op, gpu_add_op>() ||
           dsc_is_type<Op, gpu_sub_op>() ||
           dsc_is_type<Op, gpu_mul_op>();
}