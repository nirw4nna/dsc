// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#include "cpu/dsc_cpu.h"
#include "dsc_device.h"
#include "cpu/dsc_ops.h"
#include "cpu/dsc_iter.h"
#include <random>
#include <cstring> // memcpy


// ============================================================
// CPU-specific operations
//

template<typename Tx, typename To>
static DSC_INLINE void cast_op(const dsc_tensor *DSC_RESTRICT x,
                               dsc_tensor *DSC_RESTRICT out) {
    DSC_DATA(Tx, x);
    DSC_DATA(To, out);

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
    DSC_DATA(T, x);

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

    DSC_DATA(T, x);

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

template<typename T>
static DSC_INLINE void concat(dsc_tensor **to_concat,
                              const int tensors,
                              dsc_tensor *DSC_RESTRICT out,
                              const int axis_idx) {
    DSC_DATA(T, out);
    // Todo: validate the perf of this implementation
    dsc_axis_iterator *iterators = (dsc_axis_iterator *) alloca(tensors * sizeof(dsc_axis_iterator));
    for (int i = 0; i < tensors; ++i) iterators[i] = dsc_axis_iterator(to_concat[i], axis_idx);

    dsc_axis_iterator out_iterator(out, axis_idx);

    while (out_iterator.has_next()) {
        for (int i = 0; i < tensors; ++i) {
            const int axis_n = to_concat[i]->shape[axis_idx];

            T *DSC_RESTRICT src_data = (T *) to_concat[i]->buf->data;
            for (int el_idx = 0; el_idx < axis_n; ++el_idx) {
                const int index = iterators[i].index();
                out_data[out_iterator.index()] = src_data[index];

                out_iterator.next();
                iterators[i].next();
            }
        }
    }
}

void dsc_cpu_concat(dsc_device *,
                    dsc_tensor **to_concat,
                    const int tensors,
                    dsc_tensor *DSC_RESTRICT out,
                    const int axis_idx) {
    switch (out->dtype) {
        case F32:
            concat<f32>(to_concat, tensors, out, axis_idx);
            break;
        case F64:
            concat<f64>(to_concat, tensors, out, axis_idx);
            break;
        case C32:
            concat<c32>(to_concat, tensors, out, axis_idx);
            break;
        case C64:
            concat<c64>(to_concat, tensors, out, axis_idx);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }
}

// ============================================================
// Indexing and Slicing
//

template <typename T>
static DSC_INLINE void copy_slice(const dsc_tensor *DSC_RESTRICT x,
                                  dsc_tensor *DSC_RESTRICT out,
                                  const int n_slices,
                                  const dsc_slice *slices,
                                  const bool whole) {
    DSC_DATA(T, out);
    DSC_DATA(T, x);
    if (whole) {
        memcpy(out_data, x_data, out->ne * DSC_DTYPE_SIZE[out->dtype]);
    } else {
        dsc_slice_iterator x_it(x, n_slices, slices);
        dsc_for(i, out) {
            out_data[i] = x_data[x_it.index()];
            x_it.next();
        }
    }
}

void dsc_cpu_get_slice(dsc_device *,
                       const dsc_tensor *DSC_RESTRICT x,
                       dsc_tensor *DSC_RESTRICT out,
                       const int n_slices, const dsc_slice *slices,
                       const bool whole) {
    switch (out->dtype) {
        case F32:
            copy_slice<f32>(x, out, n_slices, slices, whole);
            break;
        case F64:
            copy_slice<f64>(x, out, n_slices, slices, whole);
            break;
        case C32:
            copy_slice<c32>(x, out, n_slices, slices, whole);
            break;
        case C64:
            copy_slice<c64>(x, out, n_slices, slices, whole);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }
}

template <typename T>
static DSC_INLINE void set_slice(dsc_tensor *DSC_RESTRICT xa,
                                 const bool xa_scalar,
                                 const dsc_tensor *DSC_RESTRICT xb,
                                 const bool xb_scalar,
                                 const int n_slices, const dsc_slice *slices,
                                 const bool whole) {
    DSC_DATA(T, xa);
    DSC_DATA(T, xb);
    if (xa_scalar) {
        int offset = 0;
        for (int i = 0; i < n_slices; ++i)
            offset += (slices[i].start * xa->stride[dsc_tensor_dim(xa, i)]);

        xa_data[offset] = xb_data[0];
    } else if (xb_scalar) {
        const T el = xb_data[0];
        if (whole) {
            dsc_for(i, xa) {
                xa_data[i] = el;
            }
        } else {
            for (dsc_slice_iterator xa_it(xa, n_slices, slices);
                 xa_it.has_next();
                 xa_it.next()) {
                xa_data[xa_it.index()] = el;
            }
        }
    } else {
        if (whole) {
            dsc_for(i, xa) {
                xa_data[i] = xb_data[i];
            }
        } else {
            int xb_idx = 0;
            for (dsc_slice_iterator xa_it(xa, n_slices, slices);
                 xa_it.has_next();
                 xa_it.next()) {
                xa_data[xa_it.index()] = xb_data[xb_idx++];
            }
        }
    }
}

void dsc_cpu_set_slice(dsc_device *,
                       dsc_tensor *DSC_RESTRICT xa,
                       const bool xa_scalar,
                       const dsc_tensor *DSC_RESTRICT xb,
                       const bool xb_scalar,
                       const int n_slices, const dsc_slice *slices,
                       const bool whole) {
    switch (xa->dtype) {
        case F32:
            set_slice<f32>(xa, xa_scalar, xb, xb_scalar, n_slices, slices, whole);
            break;
        case F64:
            set_slice<f64>(xa, xa_scalar, xb, xb_scalar, n_slices, slices, whole);
            break;
        case C32:
            set_slice<c32>(xa, xa_scalar, xb, xb_scalar, n_slices, slices, whole);
            break;
        case C64:
            set_slice<c64>(xa, xa_scalar, xb, xb_scalar, n_slices, slices, whole);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", xa->dtype);
    }
}

// ============================================================
// Binary Operations

template<typename T, typename Op>
static DSC_INLINE void binary_op(const dsc_tensor *xa,
                                 const dsc_tensor *xb,
                                 dsc_tensor *out,
                                 Op op) {
    DSC_DATA_ALIAS(T, xa);
    DSC_DATA_ALIAS(T, xb);
    DSC_DATA_ALIAS(T, out);

    if (dsc_is_scalar(xa)) {
        const T val = xa_data[0];
        dsc_for(i, out) {
            out_data[i] = op(
                    val,
                    xb_data[i]
            );
        }
    } else if (dsc_is_scalar(xb)) {
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
    DSC_DATA(Tx, x);
    DSC_DATA(To, out);
    
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
    DSC_DATA(T, x);
    DSC_DATA(T, out);

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

template <typename T>
static DSC_INLINE void min(const dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out,
                           const int axis_idx) {
    DSC_DATA(T, x);
    DSC_DATA(T, out);

    const int axis_n = x->shape[axis_idx];
    dsc_axis_iterator x_it(x, axis_idx, axis_n);
    dsc_for(i, out) {
        T min = dsc_inf<T, true>();
        for (int j = 0; j < axis_n; ++j) {
            min = cpu_min_op()(min, x_data[x_it.index()]);
            x_it.next();
        }
        out_data[i] = min;
    }
}

void dsc_cpu_min(dsc_device *,
                 const dsc_tensor *DSC_RESTRICT x,
                 dsc_tensor *DSC_RESTRICT out,
                 const int axis_idx) {
    switch (out->dtype) {
        case F32:
            min<f32>(x, out, axis_idx);
            break;
        case F64:
            min<f64>(x, out, axis_idx);
            break;
        case C32:
            min<c32>(x, out, axis_idx);
            break;
        case C64:
            min<c64>(x, out, axis_idx);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }
}

template <typename T>
static DSC_INLINE void max(const dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out,
                           const int axis_idx) {
    DSC_DATA(T, x);
    DSC_DATA(T, out);

    const int axis_n = x->shape[axis_idx];
    dsc_axis_iterator x_it(x, axis_idx, axis_n);
    dsc_for(i, out) {
        T max = dsc_inf<T, false>();
        for (int j = 0; j < axis_n; ++j) {
            max = cpu_max_op()(max, x_data[x_it.index()]);
            x_it.next();
        }
        out_data[i] = max;
    }
}

void dsc_cpu_max(dsc_device *,
                 const dsc_tensor *DSC_RESTRICT x,
                 dsc_tensor *DSC_RESTRICT out,
                 const int axis_idx) {
    switch (out->dtype) {
        case F32:
            max<f32>(x, out, axis_idx);
            break;
        case F64:
            max<f64>(x, out, axis_idx);
            break;
        case C32:
            max<c32>(x, out, axis_idx);
            break;
        case C64:
            max<c64>(x, out, axis_idx);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }
}