// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#include "cpu/dsc_cpu.h"
#include "dsc_device.h"
#include "cpu/dsc_ops.h"
#include "dsc_iter.h"
#include <random>


// ============================================================
// CPU-specific operations
//

template<typename Tx, typename To>
static DSC_INLINE void cast_op(const dsc_tensor *DSC_RESTRICT x,
                               dsc_tensor *DSC_RESTRICT out) {
    DSC_TENSOR_DATA(Tx, x);
    DSC_TENSOR_DATA(To, out);

    dsc_for(i, out) {
        out_data[i] = cpu_cast_op().operator()<Tx, To>(x_data[i]);
    }
}

template<typename Tx>
static DSC_INLINE void cast_op(const dsc_tensor *DSC_RESTRICT x,
                               dsc_tensor *DSC_RESTRICT out) {
    switch (out->dtype) {
        case F32:
            cast_op<Tx, f32>(x, out);
            break;
        case F64:
            cast_op<Tx, f64>(x, out);
            break;
        case C32:
            cast_op<Tx, c32>(x, out);
            break;
        case C64:
            cast_op<Tx, c64>(x, out);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

void dsc_cpu_cast(dsc_device *, const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out) {
    switch (x->dtype) {
        case F32:
            cast_op<f32>(x, out);
            break;
        case F64:
            cast_op<f64>(x, out);
            break;
        case C32:
            cast_op<c32>(x, out);
            break;
        case C64:
            cast_op<c64>(x, out);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

template<typename T>
static DSC_INLINE void assign_op(dsc_tensor *DSC_RESTRICT x,
                                 const T start, const T step) {
    DSC_TENSOR_DATA(T, x);

    T val = start;
    dsc_for(i, x) {
        x_data[i] = val;
        val = cpu_add_op()(val, step);
    }
}

void dsc_cpu_arange(dsc_device *, dsc_tensor *DSC_RESTRICT x) {
    switch (x->dtype) {
        case F32:
            assign_op<f32>(x, 0.f, 1.f);
            break;
        case F64:
            assign_op<f64>(x, 0, 1);
            break;
        case C32:
            assign_op<c32>(x,
                           dsc_complex(c32, 0.f, 0.f),
                           dsc_complex(c32, 1.f, 0.f));
            break;
        case C64:
            assign_op<c64>(x,
                           dsc_complex(c64, 0., 0.),
                           dsc_complex(c64, 1., 0.));
            break;
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

template<typename T>
static DSC_INLINE void fill_randn(dsc_tensor *DSC_RESTRICT x) {
    static_assert(dsc_is_real<T>(), "T must be real");

    DSC_TENSOR_DATA(T, x);

    std::mt19937 rng;
    std::normal_distribution<T> dist;

    dsc_for(i, x) {
        x_data[i] = dist(rng);
    }
}

void dsc_cpu_randn(dsc_device *, dsc_tensor *DSC_RESTRICT x) {
    switch (x->dtype) {
        case F32:
            fill_randn<f32>(x);
            break;
        case F64:
            fill_randn<f64>(x);
            break;
        DSC_INVALID_CASE("dtype must be real");
    }
}

// ============================================================
// Binary Operations

template<typename T, typename Op>
static DSC_INLINE void binary_op(const dsc_tensor *xa,
                                 const dsc_tensor *xb,
                                 dsc_tensor *out,
                                 Op op) {
    T *xa_data = (T *) xa->buf->data;
    T *xb_data = (T *) xb->buf->data;
    T *out_data = (T *) out->buf->data;
    const bool xa_scalar = xa->n_dim == 1 && xa->shape[dsc_tensor_dim(xa, -1)] == 1;
    const bool xb_scalar = xb->n_dim == 1 && xb->shape[dsc_tensor_dim(xb, -1)] == 1;

    if (xa_scalar) {
        const T val = xa_data[0];
        dsc_for(i, out) {
            out_data[i] = op(
                    val,
                    xb_data[i]
            );
        }
    } else if (xb_scalar) {
        const T val = xb_data[0];
        dsc_for(i, out) {
            out_data[i] = op(
                    xa_data[i],
                    val
            );
        }
    } else {
        dsc_broadcast_iterator xa_it(xa, out->shape), xb_it(xb, out->shape);
        dsc_for(i, out) {
            out_data[i] = op(
                    xa_data[xa_it.index()],
                    xb_data[xb_it.index()]
            );
            xa_it.next(), xb_it.next();
        }
    }
}

template<typename Op>
static DSC_INLINE void binary_op(const dsc_tensor *xa,
                      const dsc_tensor *xb,
                      dsc_tensor *out,
                      Op op) {
    switch (out->dtype) {
        case F32:
            binary_op<f32>(xa, xb, out, op);
            break;
        case F64:
            binary_op<f64>(xa, xb, out, op);
            break;
        case C32:
            binary_op<c32>(xa, xb, out, op);
            break;
        case C64:
            binary_op<c64>(xa, xb, out, op);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }
}

void dsc_cpu_add(dsc_device *,
                 const dsc_tensor *xa,
                 const dsc_tensor *xb,
                 dsc_tensor *out) {
    binary_op(xa, xb, out, cpu_add_op());
}

void dsc_cpu_sub(dsc_device *,
                 const dsc_tensor *xa,
                 const dsc_tensor *xb,
                 dsc_tensor *out) {
    binary_op(xa, xb, out, cpu_sub_op());
}

void dsc_cpu_mul(dsc_device *,
                 const dsc_tensor *xa,
                 const dsc_tensor *xb,
                 dsc_tensor *out) {
    binary_op(xa, xb, out, cpu_mul_op());
}

void dsc_cpu_div(dsc_device *,
                 const dsc_tensor *xa,
                 const dsc_tensor *xb,
                 dsc_tensor *out) {
    binary_op(xa, xb, out, cpu_div_op());
}

void dsc_cpu_pow(dsc_device *,
                 const dsc_tensor *xa,
                 const dsc_tensor *xb,
                 dsc_tensor *out) {
    binary_op(xa, xb, out, cpu_pow_op());
}

// ============================================================
// Unary Operations

template<typename Tx, typename To = Tx, typename Op>
static DSC_INLINE void unary_op(const dsc_tensor *DSC_RESTRICT x,
                                dsc_tensor *DSC_RESTRICT out,
                                Op op) {
    DSC_TENSOR_DATA(Tx, x);
    DSC_TENSOR_DATA(To, out);
    
    dsc_for(i, out) {
        out_data[i] = op(x_data[i]);
    }
}

template<typename Op>
static DSC_INLINE void unary_op(const dsc_tensor *DSC_RESTRICT x,
                                dsc_tensor *DSC_RESTRICT out,
                                Op op) {
    switch (x->dtype) {
        case F32:
            unary_op<f32>(x, out, op);
            break;
        case F64:
            unary_op<f64>(x, out, op);
            break;
        case C32:
            unary_op<c32>(x, out, op);
            break;
        case C64:
            unary_op<c64>(x, out, op);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

void dsc_cpu_cos(dsc_device *,
                 const dsc_tensor *DSC_RESTRICT x,
                 dsc_tensor *DSC_RESTRICT out) {
    unary_op(x, out, cpu_cos_op());
}

void dsc_cpu_sin(dsc_device *,
                 const dsc_tensor *DSC_RESTRICT x,
                 dsc_tensor *DSC_RESTRICT out) {
    unary_op(x, out, cpu_sin_op());
}

void dsc_cpu_sinc(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out) {
    unary_op(x, out, cpu_sinc_op());
}

void dsc_cpu_logn(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out) {
    unary_op(x, out, cpu_logn_op());
}

void dsc_cpu_log2(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out) {
    unary_op(x, out, cpu_log2_op());
}

void dsc_cpu_log10(dsc_device *,
                   const dsc_tensor *DSC_RESTRICT x,
                   dsc_tensor *DSC_RESTRICT out) {
    unary_op(x, out, cpu_log10_op());
}

void dsc_cpu_exp(dsc_device *,
                 const dsc_tensor *DSC_RESTRICT x,
                 dsc_tensor *DSC_RESTRICT out) {
    unary_op(x, out, cpu_exp_op());
}

void dsc_cpu_sqrt(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out) {
    unary_op(x, out, cpu_sqrt_op());
}

void dsc_cpu_i0(dsc_device *,
                const dsc_tensor *DSC_RESTRICT x,
                dsc_tensor *DSC_RESTRICT out) {
    // Explicitly instantiate only the float version of i0 otherwise
    // the is_real static assert in i0_op will trigger.
    switch (x->dtype) {
        case F32:
            unary_op<f32>(x, out, cpu_i0_op());
            break;
        case F64:
            unary_op<f64>(x, out, cpu_i0_op());
            break;
        DSC_INVALID_CASE("dtype must be real");
    }
}

void dsc_cpu_abs(dsc_device *,
                 const dsc_tensor *DSC_RESTRICT x,
                 dsc_tensor *DSC_RESTRICT out) {
    switch (x->dtype) {
        case F32:
            unary_op<f32, f32>(x, out, cpu_abs_op());
            break;
        case F64:
            unary_op<f64, f64>(x, out, cpu_abs_op());
            break;
        case C32:
            unary_op<c32, f32>(x, out, cpu_abs_op());
            break;
        case C64:
            unary_op<c64, f64>(x, out, cpu_abs_op());
            break;
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

void dsc_cpu_angle(dsc_device *,
                   const dsc_tensor *DSC_RESTRICT x,
                   dsc_tensor *DSC_RESTRICT out) {
    switch (x->dtype) {
        case F32:
            unary_op<f32, f32>(x, out, cpu_atan2_op());
            break;
        case F64:
            unary_op<f64, f64>(x, out, cpu_atan2_op());
            break;
        case C32:
            unary_op<c32, f32>(x, out, cpu_atan2_op());
            break;
        case C64:
            unary_op<c64, f64>(x, out, cpu_atan2_op());
            break;
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

void dsc_cpu_conj(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out) {
    switch (x->dtype) {
        case C32:
            unary_op<c32>(x, out, cpu_conj_op());
            break;
        case C64:
            unary_op<c64>(x, out, cpu_conj_op());
            break;
        DSC_INVALID_CASE("dtype must be complex");
    }
}

void dsc_cpu_real(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out) {
    switch (x->dtype) {
        case C32:
            unary_op<c32, f32>(x, out, cpu_real_op());
            break;
        case C64:
            unary_op<c64, f64>(x, out, cpu_real_op());
            break;
        DSC_INVALID_CASE("dtype must be complex");
    }
}

void dsc_cpu_imag(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out) {
    switch (x->dtype) {
        case F32:
            unary_op<f32, f32>(x, out, cpu_imag_op());
            break;
        case F64:
            unary_op<f64, f64>(x, out, cpu_imag_op());
            break;
        case C32:
            unary_op<c32, f32>(x, out, cpu_imag_op());
            break;
        case C64:
            unary_op<c64, f64>(x, out, cpu_imag_op());
            break;
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

void dsc_cpu_clip(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out,
                  const f64 x_min, const f64 x_max) {
    switch (out->dtype) {
        case F32:
            unary_op<f32>(x, out, cpu_clip_op((f32) x_min, (f32) x_max));
            break;
        case F64:
            unary_op<f64>(x, out, cpu_clip_op(x_min, x_max));
            break;
        case C32:
            unary_op<c32>(x, out,
                     cpu_clip_op(dsc_complex(c32, (f32) x_min, dsc_zero<f32>()),
                                 dsc_complex(c32, (f32) x_max, dsc_zero<f32>())));
            break;
        case C64:
            unary_op<c64>(x, out,
                     cpu_clip_op(dsc_complex(c64, x_min, dsc_zero<f64>()),
                                 dsc_complex(c64, x_max, dsc_zero<f64>())));
            break;
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }
}

// ============================================================
// Unary Operations Along Axis

template <typename T>
static DSC_INLINE void sum(const dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out,
                           const int axis_idx) {
    DSC_TENSOR_DATA_R(T, x);
    DSC_TENSOR_DATA_R(T, out);

    const int axis_n = x->shape[axis_idx];
    dsc_axis_iterator x_it(x, axis_idx, axis_n);
    dsc_for(i, out) {
        T acc = dsc_zero<T>();
        for (int j = 0; j < axis_n; ++j) {
            acc = cpu_add_op()(acc, x_data[x_it.index()]);
            x_it.next();
        }
        out_data[i] = acc;
    }
}

void dsc_cpu_sum(dsc_device *,
                 const dsc_tensor *DSC_RESTRICT x,
                 dsc_tensor *DSC_RESTRICT out,
                 const int axis_idx) {
    // TODO: I'd like to have a generic 'reduce' routine instead of defining each reduction explicitly
    switch (out->dtype) {
        case F32:
            sum<f32>(x, out, axis_idx);
            break;
        case F64:
            sum<f64>(x, out, axis_idx);
            break;
        case C32:
            sum<c32>(x, out, axis_idx);
            break;
        case C64:
            sum<c64>(x, out, axis_idx);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }
}
