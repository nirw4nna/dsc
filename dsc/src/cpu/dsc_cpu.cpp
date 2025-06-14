// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#include "cpu/dsc_cpu.h"
#include "cpu/dsc_blas.h"
#include "cpu/dsc_iter.h"
#include "cpu/dsc_ops.h"
#include "dsc_device.h"
#include <cstring> // memcpy
#include <random>
#include <algorithm>

// ============================================================
// CPU-specific operations
//

template<typename Tx, typename To>
static DSC_INLINE void cast_op(dsc_device *,
                               const dsc_tensor *DSC_RESTRICT x,
                               dsc_tensor *DSC_RESTRICT out) {
    DSC_DATA(Tx, x);
    DSC_DATA(To, out);

    dsc_for(i, out) {
        out_data[i] = cpu_cast_op().operator()<Tx, To>(x_data[i]);
    }
}

template<typename Tx>
static DSC_INLINE void cast_op(dsc_device *dev,
                               const dsc_tensor *DSC_RESTRICT x,
                               dsc_tensor *DSC_RESTRICT out) {
    switch (out->dtype) {
        case BOOL:
            cast_op<Tx, bool>(dev, x, out);
            break;
        case I32:
            cast_op<Tx, i32>(dev, x, out);
            break;
        case F32:
            cast_op<Tx, f32>(dev, x, out);
            break;
        case F64:
            cast_op<Tx, f64>(dev, x, out);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

void dsc_cpu_cast(dsc_device *dev,
                  const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out) {
    switch (x->dtype) {
        case BOOL:
            cast_op<bool>(dev, x, out);
            break;
        case I32:
            cast_op<i32>(dev, x, out);
            break;
        case BF16:
            cast_op<bf16>(dev, x, out);
            break;
        case F32:
            cast_op<f32>(dev, x, out);
            break;
        case F64:
            cast_op<f64>(dev, x, out);
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

void dsc_cpu_arange(dsc_device *,
                    dsc_tensor *DSC_RESTRICT x,
                    const f64 start, const f64 step) {
    switch (x->dtype) {
        case I32:
            assign_op<i32>(x, (i32) start, (i32) step);
            break;
        case F32:
            assign_op<f32>(x, (f32) start, (f32) step);
            break;
        case F64:
            assign_op<f64>(x, start, step);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

template<typename T>
static DSC_INLINE void repeat(const dsc_tensor *DSC_RESTRICT x,
                              dsc_tensor *DSC_RESTRICT out,
                              const int repeats, const int axis_idx) {
    DSC_DATA(T, x);
    DSC_DATA(T, out);

    dsc_axis_iterator x_it(x, axis_idx), out_it(out, axis_idx);
    while (x_it.has_next()) {
        const int x_idx = x_it.index();
        const T x_val = x_data[x_idx];

        for (int i = 0; i < repeats; ++i) {
            out_data[out_it.index()] = x_val;
            out_it.next();
        }

        x_it.next();
    }
}

void dsc_cpu_repeat(dsc_device *,
                    const dsc_tensor *DSC_RESTRICT x,
                    dsc_tensor *DSC_RESTRICT out,
                    const int repeats, const int axis_idx) {
    switch (x->dtype) {
        case BOOL:
            repeat<bool>(x, out, repeats, axis_idx);
            break;
        case I32:
            repeat<i32>(x, out, repeats, axis_idx);
            break;
        case F32:
            repeat<f32>(x, out, repeats, axis_idx);
            break;
        case F64:
            repeat<f64>(x, out, repeats, axis_idx);
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

template<typename T>
static DSC_INLINE void topk(const dsc_tensor *DSC_RESTRICT x,
                            dsc_tensor *DSC_RESTRICT tmp_values,
                            dsc_tensor *DSC_RESTRICT tmp_indexes,
                            dsc_tensor *DSC_RESTRICT out_values,
                            dsc_tensor *DSC_RESTRICT out_indexes,
                            const int k, const int axis_idx,
                            const bool largest) {
    DSC_DATA(T, x);
    DSC_DATA(T, tmp_values);
    DSC_DATA(i32, tmp_indexes);
    DSC_DATA(T, out_values);
    DSC_DATA(i32, out_indexes);

    const int axis_n = x->shape[axis_idx];
    dsc_axis_iterator x_it(x, axis_idx), out_it(out_values, axis_idx); // values and indexes are exactly the same for out
    while (x_it.has_next()) {
        for (int i = 0; i < axis_n; ++i) {
            const int idx = x_it.index();
            tmp_indexes_data[i] = i;
            tmp_values_data[i] = x_data[idx];
            x_it.next();
        }

        // Sort the indexes based on the values
        std::sort(tmp_indexes_data, tmp_indexes_data + axis_n, [&tmp_values_data, largest](const i32 xa_idx, const i32 xb_idx) -> bool {
            const T xa = tmp_values_data[xa_idx];
            const T xb = tmp_values_data[xb_idx];
            return largest ? xa > xb : xa < xb;
        });

        // Copy the top K elements from tmp to out
        for (int i = 0; i < k; ++i) {
            const int out_idx = out_it.index();
            const i32 val_idx = tmp_indexes_data[i];
            out_indexes_data[out_idx] = val_idx;
            out_values_data[out_idx] = tmp_values_data[val_idx];
            out_it.next();
        }
    }
}

void dsc_cpu_topk(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT tmp_values,
                  dsc_tensor *DSC_RESTRICT tmp_indexes,
                  dsc_tensor *DSC_RESTRICT out_values,
                  dsc_tensor *DSC_RESTRICT out_indexes,
                  const int k, const int axis_idx,
                  const bool largest) {
    switch (x->dtype) {
        case I32:
            topk<i32>(x, tmp_values, tmp_indexes, out_values, out_indexes, k, axis_idx, largest);
            break;
        case F32:
            topk<f32>(x, tmp_values, tmp_indexes, out_values, out_indexes, k, axis_idx, largest);
            break;
        case F64:
            topk<f64>(x, tmp_values, tmp_indexes, out_values, out_indexes, k, axis_idx, largest);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

template<typename T>
static DSC_INLINE void multinomial(const dsc_tensor *DSC_RESTRICT x,
                                   dsc_tensor *DSC_RESTRICT out,
                                   const int num_samples) {
    DSC_DATA(T, x);
    DSC_DATA(i32, out);

    const int rows = dsc_tensor_get_dim(x, -2);
    const int x_cols = dsc_tensor_get_dim(x, -1);
    const int out_cols = dsc_tensor_get_dim(out, -1);

    std::random_device rd;
    std::mt19937 rng(rd());
    for (int i = 0; i < rows; ++i) {
        // Create a discrete distribution using the probabilities (x[i, 0], ..., x[i, cols - 1])
        std::discrete_distribution<> dist(&x_data[i * x_cols], &x_data[(i+1) * x_cols]);

        // Sample num_samples values from the distribution
        for (int sample = 0; sample < num_samples; ++sample) {
            const int o = dist(rng);
            out_data[i * out_cols + sample] = o;
        }
    }
}

void dsc_cpu_multinomial(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out,
                         const int num_samples) {
    switch (x->dtype) {
        case F32:
            multinomial<f32>(x, out, num_samples);
            break;
        case F64:
            multinomial<f64>(x, out, num_samples);
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
static void binary_op(dsc_device *,
                      const dsc_tensor *xa,
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
static DSC_INLINE void binary_op(dsc_device *dev,
                                 const dsc_tensor *xa,
                                 const dsc_tensor *xb,
                                 dsc_tensor *out,
                                 Op op) {
    switch (out->dtype) {
        case BOOL:
            // Normal binary operations handle casting beforehand, in case of boolean operations
            // this is not the case: the output will always be bool but the input could be anything
            if constexpr (is_comparison_op<Op>()) {
                switch (xa->dtype) {
                    case BOOL:
                        binary_op<bool, bool>(dev, xa, xb, out, op);
                        break;
                    case I32:
                        binary_op<bool, i32>(dev, xa, xb, out, op);
                        break;
                    case F32:
                        binary_op<bool, f32>(dev, xa, xb, out, op);
                        break;
                    case F64:
                        binary_op<bool, f64>(dev, xa, xb, out, op);
                        break;
                    DSC_INVALID_CASE("unknown dtype=%d", xa->dtype);
                }
            } else if constexpr (is_bool_arith_op<Op>()) {
                // This is handled as a normal math op but the catch is that the operator is actually a
                // boolean operator
                binary_op<bool>(dev, xa, xb, out, op);
            } else {
                DSC_LOG_FATAL("invalid op");
            }
            break;
        case I32:
            binary_op<i32>(dev, xa, xb, out, op);
            break;
        case F32:
            binary_op<f32>(dev, xa, xb, out, op);
            break;
        case F64:
            binary_op<f64>(dev, xa, xb, out, op);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }
}

void dsc_cpu_add(dsc_device *dev,
                 const dsc_tensor *xa,
                 const dsc_tensor *xb,
                 dsc_tensor *out) {
    binary_op(dev, xa, xb, out, cpu_add_op());
}

void dsc_cpu_sub(dsc_device *dev,
                 const dsc_tensor *xa,
                 const dsc_tensor *xb,
                 dsc_tensor *out) {
    binary_op(dev, xa, xb, out, cpu_sub_op());
}

void dsc_cpu_mul(dsc_device *dev,
                 const dsc_tensor *xa,
                 const dsc_tensor *xb,
                 dsc_tensor *out) {
    binary_op(dev, xa, xb, out, cpu_mul_op());
}

void dsc_cpu_div(dsc_device *dev,
                 const dsc_tensor *xa,
                 const dsc_tensor *xb,
                 dsc_tensor *out) {
    binary_op(dev, xa, xb, out, cpu_div_op());
}

void dsc_cpu_pow(dsc_device *dev,
                 const dsc_tensor *xa,
                 const dsc_tensor *xb,
                 dsc_tensor *out) {
    binary_op(dev, xa, xb, out, cpu_pow_op());
}

void dsc_cpu_compare(dsc_device *dev,
                     const dsc_tensor *xa,
                     const dsc_tensor *xb,
                     const dsc_comparison_op comp,
                     dsc_tensor *out) {
    switch (comp) {
        case EQ:
            binary_op(dev, xa, xb, out, cpu_eq_op());
            break;
        case NE:
            binary_op(dev, xa, xb, out, cpu_ne_op());
            break;
        case LT:
            binary_op(dev, xa, xb, out, cpu_lt_op());
            break;
        case LE:
            binary_op(dev, xa, xb, out, cpu_le_op());
            break;
        case GT:
            binary_op(dev, xa, xb, out, cpu_gt_op());
            break;
        case GE:
            binary_op(dev, xa, xb, out, cpu_ge_op());
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
        case BOOL:
            masked_fill(x, mask, (bool) value);
            break;
        case I32:
            masked_fill(x, mask, (i32) value);
            break;
        case F32:
            masked_fill(x, mask, (f32) value);
            break;
        case F64:
            masked_fill(x, mask, value);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

template <typename T>
static DSC_INLINE void outer(const dsc_tensor *DSC_RESTRICT xa,
                             const dsc_tensor *DSC_RESTRICT xb,
                             dsc_tensor *DSC_RESTRICT out) {
    DSC_DATA(T, xa);
    DSC_DATA(T, xb);
    DSC_DATA(T, out);

    for (int i = 0; i < xa->ne; ++i) {
        for (int j = 0; j < xb->ne; ++j) {
            out_data[i * xb->ne + j] = cpu_mul_op()(xa_data[i], xb_data[j]);
        }
    }
}

void dsc_cpu_outer(dsc_device *,
                   const dsc_tensor *DSC_RESTRICT xa,
                   const dsc_tensor *DSC_RESTRICT xb,
                   dsc_tensor *DSC_RESTRICT out) {
    switch (xa->dtype) {
        case BOOL:
            outer<bool>(xa, xb, out);
            break;
        case I32:
            outer<i32>(xa, xb, out);
            break;
        case F32:
            outer<f32>(xa, xb, out);
            break;
        case F64:
            outer<f64>(xa, xb, out);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", xa->dtype);
    }
}

template<typename T>
static DSC_INLINE void where(const dsc_tensor *DSC_RESTRICT condition,
                             const dsc_tensor *DSC_RESTRICT input,
                             const dsc_tensor *DSC_RESTRICT other,
                             dsc_tensor *DSC_RESTRICT out) {
    DSC_DATA(bool, condition);
    DSC_DATA(T, input);
    DSC_DATA(T, other);
    DSC_DATA(T, out);

    const bool input_scalar = dsc_is_scalar(input);
    const bool other_scalar = dsc_is_scalar(other);

    const T in_scalar_val = input_data[0];
    const T oth_scalar_val = other_data[0];

    dsc_broadcast_iterator cond_it(condition, out->shape);

    dsc_for(i, out) {
        const int idx = cond_it.index();
        if (condition_data[idx]) {
            out_data[i] = input_scalar ? in_scalar_val : input_data[idx];
        } else {
            out_data[i] = other_scalar ? oth_scalar_val : other_data[idx];
        }
        cond_it.next();
    }
}

void dsc_cpu_where(dsc_device *,
                   const dsc_tensor *DSC_RESTRICT condition,
                   const dsc_tensor *DSC_RESTRICT input,
                   const dsc_tensor *DSC_RESTRICT other,
                   dsc_tensor *DSC_RESTRICT out) {
    switch (input->dtype) {
        case BOOL:
            where<bool>(condition, input, other, out);
            break;
        case I32:
            where<i32>(condition, input, other, out);
            break;
        case F32:
            where<f32>(condition, input, other, out);
            break;
        case F64:
            where<f64>(condition, input, other, out);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", input->dtype);
    }
}


void dsc_cpu_matmul(dsc_device *dev,
                    const dsc_tensor *DSC_RESTRICT xa,
                    const dsc_tensor *DSC_RESTRICT xb,
                    const bool trans_b,
                    dsc_tensor *DSC_RESTRICT out) {
    dsc_blas_ctx *blas_ctx = (dsc_blas_ctx *) dev->extra_info;

    const int stride_a = dsc_tensor_get_stride(xa, -2);
    const int stride_b = dsc_tensor_get_stride(xb, -2);
    const int stride_out = dsc_tensor_get_stride(out, -2);
    const int m = dsc_tensor_get_dim(out, -2);
    const int n = dsc_tensor_get_dim(out, -1);
    const int k = dsc_tensor_get_dim(xa, -1);
    // If the number of rows in the A matrix is 1 then this is a GEVM
    const bool is_gevm = m == 1;

    const int d0_out = out->shape[0];
    const int d1_out = out->shape[1];
    // We already validated the shape so if the resulting dim is != it means we need to apply broadcasting
    const int xa_stride_d0 = xa->shape[0] != d0_out ? 0 : xa->stride[0];
    const int xa_stride_d1 = xa->shape[1] != d1_out ? 0 : xa->stride[1];
    const int xb_stride_d0 = xb->shape[0] != d0_out ? 0 : xb->stride[0];
    const int xb_stride_d1 = xb->shape[1] != d1_out ? 0 : xb->stride[1];

    switch (xa->dtype) {
        case F32: {
            DSC_DATA(f32, xa);
            DSC_DATA(f32, xb);
            DSC_DATA(f32, out);

            for (int d0 = 0; d0 < d0_out; ++d0) {
                for (int d1 = 0; d1 < d1_out; ++d1) {
                    const int out_offset = d0 * out->stride[0] + d1 * out->stride[1];
                    const int xa_offset = d0 * xa_stride_d0 + d1 * xa_stride_d1;
                    const int xb_offset = d0 * xb_stride_d0 + d1 * xb_stride_d1;
                    if (trans_b) {
                        if (is_gevm) {
                            dsc_sgevm_trans(blas_ctx, n, k,
                                            &xa_data[xa_offset],
                                            &xb_data[xb_offset], stride_b,
                                            &out_data[out_offset]);
                        } else {
                            dsc_sgemm(blas_ctx, TRANS, m, n, k,
                                      &xa_data[xa_offset], stride_a,
                                      &xb_data[xb_offset], stride_b,
                                      &out_data[out_offset], stride_out);
                        }
                    } else {
                        dsc_sgemm(blas_ctx, NO_TRANS, m, n, k,
                                  &xa_data[xa_offset], stride_a,
                                  &xb_data[xb_offset], stride_b,
                                  &out_data[out_offset], stride_out);
                    }
                }
            }
            break;
        }
        case F64: {
            DSC_DATA(f64, xa);
            DSC_DATA(f64, xb);
            DSC_DATA(f64, out);

            for (int d0 = 0; d0 < d0_out; ++d0) {
                for (int d1 = 0; d1 < d1_out; ++d1) {
                    const int out_offset = d0 * out->stride[0] + d1 * out->stride[1];
                    const int xa_offset = d0 * xa_stride_d0 + d1 * xa_stride_d1;
                    const int xb_offset = d0 * xb_stride_d0 + d1 * xb_stride_d1;
                    if (trans_b) {
                        if (is_gevm) {
                            dsc_dgevm_trans(blas_ctx, n, k,
                                            &xa_data[xa_offset],
                                            &xb_data[xb_offset], stride_b,
                                            &out_data[out_offset]);
                        } else {
                            dsc_dgemm(blas_ctx, TRANS, m, n, k,
                                      &xa_data[xa_offset], stride_a,
                                      &xb_data[xb_offset], stride_b,
                                      &out_data[out_offset], stride_out);
                        }
                    } else {
                        dsc_dgemm(blas_ctx, NO_TRANS, m, n, k,
                                  &xa_data[xa_offset], stride_a,
                                  &xb_data[xb_offset], stride_b,
                                  &out_data[out_offset], stride_out);
                    }
                }
            }
            break;
        }
        DSC_INVALID_CASE("unsupported dtype=%d", xa->dtype);
    }
}

// ============================================================
// Unary Operations

template<typename Tx, typename To = Tx, typename Op>
static DSC_INLINE void unary_op(dsc_device *,
                                const dsc_tensor *DSC_RESTRICT x,
                                dsc_tensor *DSC_RESTRICT out,
                                Op op) {
    DSC_DATA(Tx, x);
    DSC_DATA(To, out);

    dsc_for(i, out) {
        out_data[i] = op(x_data[i]);
    }
}

template<typename Op>
static DSC_INLINE void unary_op(dsc_device *dev,
                                const dsc_tensor *DSC_RESTRICT x,
                                dsc_tensor *DSC_RESTRICT out,
                                Op op) {
    switch (x->dtype) {
        case F32:
            unary_op<f32>(dev, x, out, op);
            break;
        case F64:
            unary_op<f64>(dev, x, out, op);
            break;
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

void dsc_cpu_cos(dsc_device *dev,
                 const dsc_tensor *DSC_RESTRICT x,
                 dsc_tensor *DSC_RESTRICT out) {
    unary_op(dev, x, out, cpu_cos_op());
}

void dsc_cpu_sin(dsc_device *dev,
                 const dsc_tensor *DSC_RESTRICT x,
                 dsc_tensor *DSC_RESTRICT out) {
    unary_op(dev, x, out, cpu_sin_op());
}

void dsc_cpu_tanh(dsc_device *dev,
                  const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out) {
    unary_op(dev, x, out, cpu_tanh_op());
}

void dsc_cpu_exp(dsc_device *dev,
                 const dsc_tensor *DSC_RESTRICT x,
                 dsc_tensor *DSC_RESTRICT out) {
    unary_op(dev, x, out, cpu_exp_op());
}

void dsc_cpu_sqrt(dsc_device *dev,
                  const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out) {
    unary_op(dev, x, out, cpu_sqrt_op());
}

// ============================================================
// Unary Operations Along Axis

template<typename T, typename ROp>
static DSC_INLINE void reduce(const dsc_tensor *DSC_RESTRICT x,
                              dsc_tensor *DSC_RESTRICT out,
                              const int axis_idx,
                              const T initial_value,
                              ROp reduction) {
    DSC_DATA(T, out);
    DSC_DATA(T, x);

    const int axis_n = x->shape[axis_idx];
    dsc_axis_iterator x_it(x, axis_idx, axis_n);
    dsc_for(i, out) {
        T reduced_val = initial_value;
        for (int j = 0; j < axis_n; ++j) {
            reduced_val = reduction(reduced_val, x_data[x_it.index()]);
            x_it.next();
        }
        out_data[i] = reduced_val;
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