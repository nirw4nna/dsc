// Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "cuda/dsc_cuda.cuh"


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


struct cuda_cast_op {
    template<typename Tin, typename Tout>
    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE Tout operator()(const Tin in) const {
        if constexpr (dsc_is_type<Tin, bf16>()) {
            // Same as CPU...maybe should use some CUDA intrinsics?
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

struct cuda_add_op {
    template<typename T>
    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        if constexpr (dsc_is_type<T, bool>()) {
            return xa || xb;
        } else {
            return xa + xb;
        }
    }
};

struct cuda_atomic_add_op {
    template<typename T>
    DSC_CUDA_FUNC DSC_INLINE void operator()(T *x, const T val) const {
        if constexpr (dsc_is_type<T, bool>()) {
            atomicOr(x, val);
        } else {
            atomicAdd(x, val);
        }
    }
};

struct cuda_sub_op {
    template<typename T>
    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        if constexpr (dsc_is_type<T, bool>()) {
            return xa ^ xb;
        } else {
            return xa - xb;
        }
    }
};

struct cuda_mul_op {
    template<typename T>
    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        if constexpr (dsc_is_type<T, bool>()) {
            return xa && xb;
        } else {
            return xa * xb;
        }
    }
};

struct cuda_div_op {
    template<typename T>
    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        return xa / xb;
    }
};

struct cuda_pow_op {
    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE i32 operator()(const i32 base, const i32 exp) const {
        i32 acc = 1;
        for (int i = 0; i < exp; ++i) acc *= base;
        return acc;
    }

    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE f32 operator()(const f32 base, const f32 exp) const {
        return powf(base, exp);
    }

    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE f64 operator()(const f64 base, const f64 exp) const {
        return pow(base, exp);
    }
};

struct cuda_cos_op {
    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE f32 operator()(const f32 x) const {
        return cosf(x);
    }

    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE f64 operator()(const f64 x) const {
        return cos(x);
    }
};

struct cuda_sin_op {
    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE f32 operator()(const f32 x) const {
        return sinf(x);
    }

    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE f64 operator()(const f64 x) const {
        return sin(x);
    }
};

struct cuda_tanh_op {
    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE f32 operator()(const f32 x) const {
        return tanhf(x);
    }

    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE f64 operator()(const f64 x) const {
        return tanh(x);
    }
};

struct cuda_sqrt_op {
    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE f32 operator()(const f32 x) const {
        return sqrtf(x);
    }

    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE f64 operator()(const f64 x) const {
        return sqrt(x);
    }
};

struct cuda_exp_op {
    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE f32 operator()(const f32 x) const {
        return expf(x);
    }

    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE f64 operator()(const f64 x) const {
        return exp(x);
    }
};

struct cuda_max_op {
    template<typename T>
    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        return DSC_MAX(xa, xb);
    }
};

struct cuda_atomic_max_op {
    DSC_CUDA_FUNC DSC_INLINE void operator()(f32 *x, const f32 val) const {
        atomic_cas_f32(x, DSC_MAX(val, assumed_val));
    }

    DSC_CUDA_FUNC DSC_INLINE void operator()(f64 *x, const f64 val) const {
        atomic_cas_f64(x, DSC_MAX(val, assumed_val));
    }
};

struct cuda_min_op {
    template<typename T>
    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE T operator()(const T xa, const T xb) const {
        return DSC_MIN(xa, xb);
    }
};

struct cuda_atomic_min_op {
    DSC_CUDA_FUNC DSC_INLINE void operator()(f32 *x, const f32 val) const {
        atomic_cas_f32(x, DSC_MIN(val, assumed_val));
    }

    DSC_CUDA_FUNC DSC_INLINE void operator()(f64 *x, const f64 val) const {
        atomic_cas_f64(x, DSC_MIN(val, assumed_val));
    }
};

struct cuda_eq_op {
    template<typename T>
    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE bool operator()(const T xa, const T xb) const {
        return xa == xb;
    }
};

struct cuda_ne_op {
    template<typename T>
    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE bool operator()(const T xa, const T xb) const {
        return !cuda_eq_op()(xa, xb);
    }
};

struct cuda_lt_op {
    template<typename T>
    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE bool operator()(const T xa, const T xb) const {
        return xa < xb;
    }
};
struct cuda_le_op {
    template<typename T>
    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE bool operator()(const T xa, const T xb) const {
        return xa <= xb;
    }
};

struct cuda_gt_op {
    template<typename T>
    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE bool operator()(const T xa, const T xb) const {
        return xa > xb;
    }
};

struct cuda_ge_op {
    template<typename T>
    DSC_CUDA_FUNC DSC_INLINE DSC_STRICTLY_PURE bool operator()(const T xa, const T xb) const {
        return xa >= xb;
    }
};


template<typename Op>
consteval bool is_comparison_op() {
    return dsc_is_type<Op, cuda_eq_op>() ||
           dsc_is_type<Op, cuda_ne_op>() ||
           dsc_is_type<Op, cuda_lt_op>() ||
           dsc_is_type<Op, cuda_le_op>() ||
           dsc_is_type<Op, cuda_gt_op>() ||
           dsc_is_type<Op, cuda_ge_op>();
}

template<typename Op>
consteval bool is_bool_arith_op() {
    return dsc_is_type<Op, cuda_add_op>() ||
           dsc_is_type<Op, cuda_sub_op>() ||
           dsc_is_type<Op, cuda_mul_op>();
}