// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#include "cpu/dsc_cpu.h"
#include "cpu/dsc_gemm.h"
#include "cpu/dsc_iter.h"
#include "cpu/dsc_ops.h"
#include "dsc_device.h"
#include <cpu/dsc_gemm.h>
#include <cstring>// memcpy
#include <random>


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
        case BOOL:
            cast_op<Tx, bool>(x, out);
            break;
        case I32:
            cast_op<Tx, i32>(x, out);
            break;
        case F32:
            cast_op<Tx, f32>(x, out);
            break;
        case F64:
            cast_op<Tx, f64>(x, out);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

void dsc_cpu_cast(dsc_device *, const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out) {
    switch (x->dtype) {
        case BOOL:
            cast_op<bool>(x, out);
            break;
        case I32:
            cast_op<i32>(x, out);
            break;
        case F32:
            cast_op<f32>(x, out);
            break;
        case F64:
            cast_op<f64>(x, out);
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
        case I32:
            assign_op<i32>(x, 0, 1);
            break;
        case F32:
            assign_op<f32>(x, 0.f, 1.f);
            break;
        case F64:
            assign_op<f64>(x, 0, 1);
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
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

// ============================================================
// Tensor Manipulation

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
        case BOOL:
            concat<bool>(to_concat, tensors, out, axis_idx);
            break;
        case I32:
            concat<i32>(to_concat, tensors, out, axis_idx);
            break;
        case F32:
            concat<f32>(to_concat, tensors, out, axis_idx);
            break;
        case F64:
            concat<f64>(to_concat, tensors, out, axis_idx);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }
}

template<typename T>
static DSC_INLINE void split(const dsc_tensor *DSC_RESTRICT x,
                             dsc_tensor *DSC_RESTRICT out,
                             const int axis_idx, const int offset) {
    DSC_DATA(T, x);
    DSC_DATA(T, out);

    dsc_axis_iterator x_it(x, axis_idx, -1, offset);
    dsc_for(i, out) {
        out_data[i] = x_data[x_it.index()];
        x_it.next();
    }
}

void dsc_cpu_split(dsc_device *,
                   const dsc_tensor *DSC_RESTRICT x,
                   dsc_tensor *DSC_RESTRICT out,
                   const int axis_idx, const int offset) {
    switch (out->dtype) {
        case BOOL:
            split<bool>(x, out, axis_idx, offset);
            break;
        case I32:
            split<i32>(x, out, axis_idx, offset);
            break;
        case F32:
            split<f32>(x, out, axis_idx, offset);
            break;
        case F64:
            split<f64>(x, out, axis_idx, offset);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }
}

template <typename T>
static DSC_INLINE void transpose(const dsc_tensor *DSC_RESTRICT x,
                                 dsc_tensor *DSC_RESTRICT out,
                                 const int *new_shape,
                                 const int *new_stride) {
    DSC_DATA(T, x);
    DSC_DATA(T, out);

    dsc_broadcast_iterator x_it(new_shape, new_stride);
    dsc_for(i, out) {
        out_data[i] = x_data[x_it.index()];
        x_it.next();
    }
}

void dsc_cpu_transpose(dsc_device *,
                       const dsc_tensor *DSC_RESTRICT x,
                       dsc_tensor *DSC_RESTRICT out,
                       const int *new_shape,
                       const int *new_stride) {
    switch (x->dtype) {
        case BOOL:
            transpose<bool>(x, out, new_shape, new_stride);
            break;
        case I32:
            transpose<i32>(x, out, new_shape, new_stride);
            break;
        case F32:
            transpose<f32>(x, out, new_shape, new_stride);
            break;
        case F64:
            transpose<f64>(x, out, new_shape, new_stride);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

template<typename T>
static DSC_INLINE void tril(const dsc_tensor *DSC_RESTRICT x,
                            const int diagonal,
                            dsc_tensor *DSC_RESTRICT out) {
    DSC_DATA(T, x);
    DSC_DATA(T, out);

    for (dsc_axis_iterator x_it(x, dsc_tensor_dim_idx(x, -1));
         x_it.has_next();
         x_it.next()) {
        const int idx = x_it.index();
        const int row = x_it.pos(-2);
        const int col = x_it.pos(-1);

        out_data[idx] = (col > (row + diagonal)) ? dsc_zero<T>() : x_data[idx];
    }
}

void dsc_cpu_tril(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT x,
                  const int diagonal,
                  dsc_tensor *DSC_RESTRICT out) {
    switch (x->dtype) {
        case BOOL:
            tril<bool>(x, diagonal, out);
            break;
        case I32:
            tril<i32>(x, diagonal, out);
            break;
        case F32:
            tril<f32>(x, diagonal, out);
            break;
        case F64:
            tril<f64>(x, diagonal, out);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
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
        case BOOL:
            copy_slice<bool>(x, out, n_slices, slices, whole);
            break;
        case I32:
            copy_slice<i32>(x, out, n_slices, slices, whole);
            break;
        case F32:
            copy_slice<f32>(x, out, n_slices, slices, whole);
            break;
        case F64:
            copy_slice<f64>(x, out, n_slices, slices, whole);
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
            offset += (slices[i].start * dsc_tensor_get_stride(xa, i));

        xa_data[offset] = xb_data[0];
    } else if (xb_scalar) {
        const T el = xb_data[0];
        if (whole) {
            // Note: this approach is very slow. Even when calling from Python with wrapping and what else
            // most of the time (90%+) is spent for this loop. At some point it will be worth investigating
            // how to speed this up.
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
        case BOOL:
            set_slice<bool>(xa, xa_scalar, xb, xb_scalar, n_slices, slices, whole);
            break;
        case I32:
            set_slice<i32>(xa, xa_scalar, xb, xb_scalar, n_slices, slices, whole);
            break;
        case F32:
            set_slice<f32>(xa, xa_scalar, xb, xb_scalar, n_slices, slices, whole);
            break;
        case F64:
            set_slice<f64>(xa, xa_scalar, xb, xb_scalar, n_slices, slices, whole);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", xa->dtype);
    }
}

// ============================================================
// Binary Operations

template<typename Tout, typename Tx = Tout, typename Op>
static DSC_INLINE void binary_op(const dsc_tensor *xa,
                                 const dsc_tensor *xb,
                                 dsc_tensor *out,
                                 Op op) {
    DSC_DATA_ALIAS(Tx, xa);
    DSC_DATA_ALIAS(Tx, xb);
    DSC_DATA_ALIAS(Tout, out);

    if (dsc_is_scalar(xa)) {
        const Tx val = xa_data[0];
        dsc_for(i, out) {
            out_data[i] = op(
                    val,
                    xb_data[i]
            );
        }
    } else if (dsc_is_scalar(xb)) {
        const Tx val = xb_data[0];
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
        case BOOL:
            // Normal binary operations handle casting beforehand, in case of boolean operations
            // this is not the case: the output will always be bool but the input could be anything
            if constexpr (is_comparison_op<Op>()) {
                switch (xa->dtype) {
                    case I32:
                        binary_op<bool, i32>(xa, xb, out, op);
                        break;
                    case F32:
                        binary_op<bool, f32>(xa, xb, out, op);
                        break;
                    case F64:
                        binary_op<bool, f64>(xa, xb, out, op);
                        break;
                    DSC_INVALID_CASE("unknown dtype=%d", xa->dtype);
                }
            } else if constexpr (is_bool_arith_op<Op>()) {
                // This is handled as a normal math op but the catch is that the operator is actually a
                // boolean operator
                binary_op<bool>(xa, xb, out, op);
            } else {
                DSC_LOG_FATAL("invalid op");
            }
            break;
        case I32:
            binary_op<i32>(xa, xb, out, op);
            break;
        case F32:
            binary_op<f32>(xa, xb, out, op);
            break;
        case F64:
            binary_op<f64>(xa, xb, out, op);
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

void dsc_cpu_compare(dsc_device *,
                     const dsc_tensor *xa,
                     const dsc_tensor *xb,
                     const dsc_comparison_op comp,
                     dsc_tensor *out) {
    switch (comp) {
        case EQ:
            binary_op(xa, xb, out, cpu_eq_op());
            break;
        case NE:
            binary_op(xa, xb, out, cpu_ne_op());
            break;
        case LT:
            binary_op(xa, xb, out, cpu_lt_op());
            break;
        case LE:
            binary_op(xa, xb, out, cpu_le_op());
            break;
        case GT:
            binary_op(xa, xb, out, cpu_gt_op());
            break;
        case GE:
            binary_op(xa, xb, out, cpu_ge_op());
            break;
        DSC_INVALID_CASE("unknown comparison=%d", comp);
    }
}

template<typename T>
static DSC_INLINE void masked_fill(dsc_tensor *x,
                                   const dsc_tensor *mask,
                                   const T value) {
    DSC_DATA(T, x);
    DSC_DATA(bool, mask);

    dsc_broadcast_iterator mask_it(mask, x->shape);
    dsc_for(i, x) {
        x_data[i] = mask_data[mask_it.index()] ? value : x_data[i];
        mask_it.next();
    }
}

void dsc_cpu_masked_fill(dsc_device *,
                         dsc_tensor *x,
                         const dsc_tensor *mask,
                         const f64 value) {
    switch (x->dtype) {
        case F32:
            masked_fill(x, mask, (f32) value);
            break;
        case F64:
            masked_fill(x, mask, value);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

void dsc_cpu_matmul(dsc_device *dev,
                    const dsc_tensor *DSC_RESTRICT xa,
                    const dsc_tensor *DSC_RESTRICT xb,
                    dsc_tensor *DSC_RESTRICT out) {
    const int stride_a = dsc_tensor_get_stride(xa, -2);
    const int stride_b = dsc_tensor_get_stride(xb, -2);
    const int stride_out = dsc_tensor_get_stride(out, -2);
    const int m = dsc_tensor_get_dim(out, -2);
    const int n = dsc_tensor_get_dim(out, -1);
    const int k = dsc_tensor_get_dim(xa, -1);

    const int d0_out = out->shape[0];
    const int d1_out = out->shape[1];
    // We already validated the shape so if the resulting dim is != it means we need to apply broadcasting
    const int xa_stride_d0 = xa->shape[0] != d0_out ? 0 : xa->stride[0];
    const int xa_stride_d1 = xa->shape[1] != d1_out ? 0 : xa->stride[1];
    const int xb_stride_d0 = xb->shape[0] != d0_out ? 0 : xb->stride[0];
    const int xb_stride_d1 = xb->shape[1] != d1_out ? 0 : xb->stride[1];

    // Packed buffers
    dsc_data_buffer *xa_buf;
    dsc_data_buffer *xb_buf;
    switch (xa->dtype) {
        case F32: {
            DSC_DATA(f32, xa);
            DSC_DATA(f32, xb);
            DSC_DATA(f32, out);
            xa_buf = dsc_data_alloc(dev, dsc_packed_A_size<f32>());
            xb_buf = dsc_data_alloc(dev, dsc_packed_B_size<f32>());

            f32 *DSC_RESTRICT packed_a = (f32 *) xa_buf->data;
            f32 *DSC_RESTRICT packed_b = (f32 *) xb_buf->data;

            for (int d0 = 0; d0 < d0_out; ++d0) {
                for (int d1 = 0; d1 < d1_out; ++d1) {
                    const int out_offset = d0 * out->stride[0] + d1 * out->stride[1];
                    const int xa_offset = d0 * xa_stride_d0 + d1 * xa_stride_d1;
                    const int xb_offset = d0 * xb_stride_d0 + d1 * xb_stride_d1;
                    dsc_gemm<f32>(m, n, k,
                                  &xa_data[xa_offset], stride_a, packed_a,
                                  &xb_data[xb_offset], stride_b, packed_b,
                                  &out_data[out_offset], stride_out);
                }
            }
            break;
        }
        case F64: {
            DSC_DATA(f64, xa);
            DSC_DATA(f64, xb);
            DSC_DATA(f64, out);

            xa_buf = dsc_data_alloc(dev, dsc_packed_A_size<f64>());
            xb_buf = dsc_data_alloc(dev, dsc_packed_B_size<f64>());

            f64 *DSC_RESTRICT packed_a = (f64 *) xa_buf->data;
            f64 *DSC_RESTRICT packed_b = (f64 *) xb_buf->data;

            for (int d0 = 0; d0 < d0_out; ++d0) {
                for (int d1 = 0; d1 < d1_out; ++d1) {
                    const int out_offset = d0 * out->stride[0] + d1 * out->stride[1];
                    const int xa_offset = d0 * xa_stride_d0 + d1 * xa_stride_d1;
                    const int xb_offset = d0 * xb_stride_d0 + d1 * xb_stride_d1;
                    dsc_gemm<f64>(m, n, k,
                                  &xa_data[xa_offset], stride_a, packed_a,
                                  &xb_data[xb_offset], stride_b, packed_b,
                                  &out_data[out_offset], stride_out);
                }
            }
            break;
        }
        DSC_INVALID_CASE("unsupported dtype=%d", xa->dtype);
    }

    dsc_data_free(dev, xa_buf);
    dsc_data_free(dev, xb_buf);
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

void dsc_cpu_tanh(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out) {
    unary_op(x, out, cpu_tanh_op());
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

// ============================================================
// Unary Operations Along Axis

template<typename T, typename ROp>
static DSC_INLINE void reduce(const dsc_tensor *DSC_RESTRICT x,
                              dsc_tensor *DSC_RESTRICT out,
                              const int axis_idx,
                              T initial_value,
                              ROp reduction) {
    DSC_DATA(T, out);
    DSC_DATA(T, x);

    const int axis_n = x->shape[axis_idx];
    dsc_axis_iterator x_it(x, axis_idx, axis_n);
    dsc_for(i, out) {
        for (int j = 0; j < axis_n; ++j) {
            initial_value = reduction(initial_value, x_data[x_it.index()]);
            x_it.next();
        }
        out_data[i] = initial_value;
    }
}

void dsc_cpu_sum(dsc_device *,
                 const dsc_tensor *DSC_RESTRICT x,
                 dsc_tensor *DSC_RESTRICT out,
                 const int axis_idx) {
    switch (out->dtype) {
        case F32:
            reduce(x, out, axis_idx, dsc_zero<f32>(), cpu_add_op());
            break;
        case F64:
            reduce(x, out, axis_idx, dsc_zero<f64>(), cpu_add_op());
            break;
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }
}

void dsc_cpu_min(dsc_device *,
                 const dsc_tensor *DSC_RESTRICT x,
                 dsc_tensor *DSC_RESTRICT out,
                 const int axis_idx) {
    switch (out->dtype) {
        case F32:
            reduce(x, out, axis_idx, dsc_inf<f32, true>(), cpu_min_op());
            break;
        case F64:
            reduce(x, out, axis_idx, dsc_inf<f64, true>(), cpu_min_op());
            break;
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }
}

void dsc_cpu_max(dsc_device *,
                 const dsc_tensor *DSC_RESTRICT x,
                 dsc_tensor *DSC_RESTRICT out,
                 const int axis_idx) {
    switch (out->dtype) {
        case F32:
            reduce(x, out, axis_idx, dsc_inf<f32, false>(), cpu_max_op());
            break;
        case F64:
            reduce(x, out, axis_idx, dsc_inf<f64, false>(), cpu_max_op());
            break;
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }
}