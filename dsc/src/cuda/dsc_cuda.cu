// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#include "cuda/dsc_cuda.h"
#include "cuda/dsc_ops.h"
#include "dsc_device.h"

#define init_slice_idx(ARR, SLICES) \
    int ARR[DSC_MAX_DIMS]{};        \
    for (int i__ = 0; i__ < DSC_MAX_DIMS; ++i__) ARR[i__] = SLICES[i__].start

template<typename Tx, typename To>
static DSC_CUDA_KERNEL void k_cast_op(const Tx *DSC_RESTRICT x,
                                      To *DSC_RESTRICT out, const int n) {
    DSC_CUDA_TID();
    DSC_CUDA_STRIDE();

    for (int i = tid; i < n; i += stride) {
        out[i] = cuda_cast_op().operator()<Tx, To>(x[i]);
    }
}

template<typename Tx>
static DSC_INLINE void cast_op(const dsc_tensor *DSC_RESTRICT x,
                               dsc_tensor *DSC_RESTRICT out) {
    DSC_TENSOR_DATA(Tx, x);
    const int n = x->ne;
    switch (out->dtype) {
        case F32: {
            DSC_TENSOR_DATA(f32, out);
            k_cast_op<Tx, f32><<<DSC_CUDA_BLOCKS(n),
                                 DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n);
            break;
        }
        case F64: {
            DSC_TENSOR_DATA(f64, out);
            k_cast_op<Tx, f64><<<DSC_CUDA_BLOCKS(n),
                                 DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n);
            break;
        }
        case C32: {
            DSC_TENSOR_DATA(c32, out);
            k_cast_op<Tx, c32><<<DSC_CUDA_BLOCKS(n),
                                 DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n);
            break;
        }
        case C64: {
            DSC_TENSOR_DATA(c64, out);
            k_cast_op<Tx, c64><<<DSC_CUDA_BLOCKS(n),
                                 DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n);
            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

void dsc_cuda_cast(dsc_device *, const dsc_tensor *DSC_RESTRICT x,
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

template <typename T>
static DSC_CUDA_KERNEL void k_assign_op(T *DSC_RESTRICT x, const int n,
                                        const T start, const T step) {
    DSC_CUDA_TID();
    DSC_CUDA_STRIDE();

    for (int i = tid; i < n; i += stride) {
        x[i] = cuda_add_op()(start, cuda_mul_op()(step, (T) i));
    }
}

void dsc_cuda_arange(dsc_device *, dsc_tensor *DSC_RESTRICT x) {
    const int n = x->ne;
    switch (x->dtype) {
        case F32: {
            DSC_TENSOR_DATA(f32, x);
            k_assign_op<f32><<<DSC_CUDA_BLOCKS(n),
                               DSC_CUDA_DEFAULT_THREADS>>>(x_data, n, 0.f, 1.f);
            break;
        }
        case F64: {
            DSC_TENSOR_DATA(f64, x);
            k_assign_op<f64><<<DSC_CUDA_BLOCKS(n),
                               DSC_CUDA_DEFAULT_THREADS>>>(x_data, n, 0., 1.);
            break;
        }
        case C32: {
            DSC_TENSOR_DATA(c32, x);
            k_assign_op<c32><<<DSC_CUDA_BLOCKS(n),
                               DSC_CUDA_DEFAULT_THREADS>>>(x_data,
                                                           n,
                                                           dsc_complex(c32, 0.f, 0.f),
                                                           dsc_complex(c32, 1.f, 0.f));
            break;
        }
        case C64: {
            DSC_TENSOR_DATA(c64, x);
            k_assign_op<c64><<<DSC_CUDA_BLOCKS(n),
                               DSC_CUDA_DEFAULT_THREADS>>>(x_data,
                                                           n,
                                                           dsc_complex(c64, 0., 0.),
                                                           dsc_complex(c64, 1., 0.));
            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

template <typename T>
static DSC_CUDA_KERNEL void k_randn(curandState *state,
                                    T *DSC_RESTRICT x,
                                    const int n) {
    DSC_CUDA_TID();
    DSC_CUDA_STRIDE();

    curandState s = state[tid];

    for (int i = tid; i < n; i += stride) {
        if constexpr (dsc_is_type<T, f32>()) {
            x[i] = curand_normal(&s);
        } else if constexpr (dsc_is_type<T, f64>()) {
            x[i] = curand_normal_double(&s);
        } else {
            static_assert("k_randn - dtype must be real");
        }
    }

    state[tid] = s;
}

void dsc_cuda_randn(dsc_device *dev, dsc_tensor *DSC_RESTRICT x) {
    dsc_cuda_dev_info *info = (dsc_cuda_dev_info *) dev->extra_info;

    switch (x->dtype) {
        case F32: {
            DSC_TENSOR_DATA(f32, x);
            k_randn<f32><<<1, DSC_CUDA_DEFAULT_THREADS>>>(info->randState, x_data, x->ne);
            break;
        }
        case F64: {
            DSC_TENSOR_DATA(f64, x);
            k_randn<f64><<<1, DSC_CUDA_DEFAULT_THREADS>>>(info->randState, x_data, x->ne);
            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

// ============================================================
// Indexing and Slicing

struct slicing_params {
    dsc_slice slices[DSC_MAX_DIMS]{};
    int shape[DSC_MAX_DIMS]{};
    int stride[DSC_MAX_DIMS]{};
};

static DSC_INLINE DSC_CUDA_FUNC int compute_linear_idx(const int *idx_arr,
                                                       const int *stride) {
    int linear_idx = 0;
    for (int i = 0; i < DSC_MAX_DIMS; ++i) {
        linear_idx += idx_arr[i] * stride[i];
    }
    return linear_idx;
}

static DSC_INLINE DSC_CUDA_FUNC void compute_idx_from_linear(int *idx_arr, int linear_idx,
                                                             const int *shape) {
    for (int i = DSC_MAX_DIMS - 1; i >= 0; i--) {
        idx_arr[i] = linear_idx % shape[i];
        linear_idx /= shape[i];
    }
}

static DSC_INLINE DSC_CUDA_FUNC void compute_slice_from_linear(int *idx_arr, int linear_idx,
                                                               const dsc_slice *slices,
                                                               const int *shape) {
    for (int i = DSC_MAX_DIMS - 1; i >= 0 && linear_idx > 0; i--) {
        const int start = slices[i].start;
        const int stop = slices[i].stop;
        const int step = slices[i].step;

        const int pos = linear_idx % shape[i];
        idx_arr[i] += pos * step;
        // Check if we rolled over
        if ((step > 0 && idx_arr[i] >= stop) ||
            (step < 0 && idx_arr[i] <= stop)) {
            // Check how many times we rolled over the i dimension
            int rollovers;
            if (step > 0) {
                const int ne = (stop - start) * step;
                rollovers = (idx_arr[i] - start) / ne;
                idx_arr[i] = start + (idx_arr[i] - start) % ne;
            } else {
                const int ne = (start - stop) * -step;
                rollovers = (start - idx_arr[i]) / ne;
                idx_arr[i] = start - (start - idx_arr[i]) % ne;
            }

            if (rollovers > 0 && i > 0)
                idx_arr[i - 1] += slices[i - 1].step * rollovers;
        }
        linear_idx /= shape[i];
    }
}

template<typename T>
static DSC_CUDA_KERNEL void k_get_slice(const T *DSC_RESTRICT x,
                                        T *DSC_RESTRICT out,
                                        const int n,
                                        const slicing_params params) {
    DSC_CUDA_TID();
    DSC_CUDA_STRIDE();

    if (tid >= n) return;

    init_slice_idx(idx_arr, params.slices);

    for (int i = tid; i < n; i += stride) {
        compute_slice_from_linear(idx_arr, i, params.slices, params.shape);
        out[i] = x[compute_linear_idx(idx_arr, params.stride)];
    }
}

static DSC_INLINE void set_slice_params(const dsc_tensor *DSC_RESTRICT x,
                                        const int n_slices,
                                        const dsc_slice *slices,
                                        slicing_params *params) {
    for (int i = 0; i < x->n_dim; ++i) {
        const int dim = dsc_tensor_dim(x, i);
        if (i < n_slices) {
            params->slices[dim] = slices[i];
        } else {
            params->slices[dim] = {0, x->shape[dim], 1};
        }
    }
}

void dsc_cuda_get_slice(dsc_device *,
                        const dsc_tensor *DSC_RESTRICT x,
                        dsc_tensor *DSC_RESTRICT out,
                        const int n_slices, const dsc_slice *slices,
                        const bool whole) {
    if (whole) {
        DSC_TENSOR_DATA_R(void, x);
        DSC_TENSOR_DATA_R(void, out);
        cudaMemcpy(out_data, x_data, out->ne * DSC_DTYPE_SIZE[out->dtype], cudaMemcpyDeviceToDevice);
        return;
    }

    slicing_params params{};
    memcpy(params.shape, out->shape, DSC_MAX_DIMS * sizeof(*out->shape));
    memcpy(params.stride, x->stride, DSC_MAX_DIMS * sizeof(*x->stride));
    set_slice_params(x, n_slices, slices, &params);

    const int n = out->ne;
    switch (out->dtype) {
        case F32: {
            DSC_TENSOR_DATA_R(f32, x);
            DSC_TENSOR_DATA_R(f32, out);
            k_get_slice<f32><<<DSC_CUDA_BLOCKS(n),
                               DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, params);
            break;
        }
        case F64: {
            DSC_TENSOR_DATA_R(f64, x);
            DSC_TENSOR_DATA_R(f64, out);
            k_get_slice<f64><<<DSC_CUDA_BLOCKS(n),
                               DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, params);
            break;
        }
        case C32: {
            DSC_TENSOR_DATA_R(c32, x);
            DSC_TENSOR_DATA_R(c32, out);
            k_get_slice<c32><<<DSC_CUDA_BLOCKS(n),
                               DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, params);
            break;
        }
        case C64: {
            DSC_TENSOR_DATA_R(c64, x);
            DSC_TENSOR_DATA_R(c64, out);
            k_get_slice<c64><<<DSC_CUDA_BLOCKS(n),
                               DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, params);
            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }
}

template<typename T>
static DSC_CUDA_KERNEL void k_set_slice(T *DSC_RESTRICT xa,
                                        const T *DSC_RESTRICT xb,
                                        const bool xb_scalar,
                                        const int n,
                                        const slicing_params params,
                                        const bool whole) {
    DSC_CUDA_TID();
    DSC_CUDA_STRIDE();

    if (tid >= n) return;

    if (whole) {
        for (int i = tid; i < n; i += stride) {
            xa[i] = xb_scalar ? xb[0] : xb[i];
        }
    } else {
        init_slice_idx(idx_arr, params.slices);

        for (int i = tid; i < n; i += stride) {
            compute_slice_from_linear(idx_arr, i, params.slices, params.shape);
            xa[compute_linear_idx(idx_arr, params.stride)] = xb_scalar ? xb[0] : xb[i];
        }
    }
}

void dsc_cuda_set_slice(dsc_device *,
                        dsc_tensor *DSC_RESTRICT xa,
                        const bool xa_scalar,
                        const dsc_tensor *DSC_RESTRICT xb,
                        const bool xb_scalar,
                        const int n_slices, const dsc_slice *slices,
                        const bool whole) {
    if (xa_scalar) {
        int offset = 0;
        for (int i = 0; i < n_slices; ++i)
            offset += (slices[i].start * xa->stride[dsc_tensor_dim(xa, i)]);

        DSC_TENSOR_DATA_R(byte, xa);
        DSC_TENSOR_DATA_R(void, xb);
        cudaMemcpy(xa_data + (offset * DSC_DTYPE_SIZE[xa->dtype]),
                    xb_data, DSC_DTYPE_SIZE[xa->dtype], cudaMemcpyDeviceToDevice);
    } else {
        slicing_params params{};
        memcpy(params.stride, xa->stride, DSC_MAX_DIMS * sizeof(*xa->stride));
        set_slice_params(xa, n_slices, slices, &params);

        int n = 1;
        for (int i = 0; i < xa->n_dim; ++i) {
            const int dim = dsc_tensor_dim(xa, i);
            const int ne_i = abs(params.slices[dim].start - params.slices[dim].stop);
            const int step_i = abs(params.slices[dim].step);
            params.shape[dim] = (ne_i + step_i - 1) / step_i;
            n *= params.shape[dim];
        }

        switch (xa->dtype) {
            case F32: {
                DSC_TENSOR_DATA_R(f32, xa);
                DSC_TENSOR_DATA_R(f32, xb);
                k_set_slice<f32><<<DSC_CUDA_BLOCKS(n),
                                   DSC_CUDA_DEFAULT_THREADS>>>(xa_data, xb_data, xb_scalar, n, params, whole);
                break;
            }
            case F64: {
                DSC_TENSOR_DATA_R(f64, xa);
                DSC_TENSOR_DATA_R(f64, xb);
                k_set_slice<f64><<<DSC_CUDA_BLOCKS(n),
                                   DSC_CUDA_DEFAULT_THREADS>>>(xa_data, xb_data, xb_scalar, n, params, whole);
                break;
            }
            case C32: {
                DSC_TENSOR_DATA_R(c32, xa);
                DSC_TENSOR_DATA_R(c32, xb);
                k_set_slice<c32><<<DSC_CUDA_BLOCKS(n),
                                   DSC_CUDA_DEFAULT_THREADS>>>(xa_data, xb_data, xb_scalar, n, params, whole);
                break;
            }
            case C64: {
                DSC_TENSOR_DATA_R(c64, xa);
                DSC_TENSOR_DATA_R(c64, xb);
                k_set_slice<c64><<<DSC_CUDA_BLOCKS(n),
                                   DSC_CUDA_DEFAULT_THREADS>>>(xa_data, xb_data, xb_scalar, n, params, whole);
                break;
            }
            DSC_INVALID_CASE("unknown dtype=%d", xa->dtype);
        }
    }
}

// ============================================================
// Binary Operations

struct binary_params {
    int out_shape[DSC_MAX_DIMS]{};
    int xa_stride[DSC_MAX_DIMS]{}, xb_stride[DSC_MAX_DIMS]{};
    int n_dim;
};

template<typename T, typename Op,
         bool xa_scalar, bool xb_scalar, bool shape_matches>
static DSC_CUDA_KERNEL void k_binary_op(const T *xa, const T *xb, T *out,
                                        const int n, Op op,
                                        const binary_params params = {}) {
    DSC_CUDA_TID();
    DSC_CUDA_STRIDE();

    if (tid >= n) return;

    if constexpr (!xa_scalar && !xb_scalar && !shape_matches) {
        for (int i = tid; i < n; i += stride) {
            int flat_idx = i;
            int xa_offset = 0, xb_offset = 0;

            for (int idx = DSC_MAX_DIMS - 1; idx >= DSC_MAX_DIMS - params.n_dim; --idx) {
                const int dim_idx = flat_idx % params.out_shape[idx];
                flat_idx /= params.out_shape[idx];

                xa_offset += dim_idx * params.xa_stride[idx];
                xb_offset += dim_idx * params.xb_stride[idx];
            }

            out[i] = op(xa[xa_offset], xb[xb_offset]);
        }
    } else {
        for (int i = tid; i < n; i += stride) {
            if constexpr (xa_scalar) {
                out[i] = op(xa[0], xb[i]);
            } else if constexpr (xb_scalar) {
                out[i] = op(xa[i], xb[0]);
            } else if constexpr (shape_matches) {
                out[i] = op(xa[i], xb[i]);
            }
        }
    }
}

template<typename T, typename Op>
static DSC_INLINE void binary_op(const dsc_tensor *xa,
                                 const dsc_tensor *xb,
                                 dsc_tensor *out,
                                 Op op) {
    const bool xa_scalar = dsc_is_scalar(xa);
    const bool xb_scalar = dsc_is_scalar(xb);
    const bool same_shape = xa->n_dim == xb->n_dim && memcmp(&xa->shape[dsc_tensor_dim(xa, 0)],
                                                             &xb->shape[dsc_tensor_dim(xb, 0)],
                                                             xa->n_dim * sizeof(*xa->shape)) == 0;
    DSC_TENSOR_DATA(T, xa);
    DSC_TENSOR_DATA(T, xb);
    DSC_TENSOR_DATA(T, out);

    const int n = out->ne;

    if (xa_scalar) {
        k_binary_op<T, Op, true, false, false><<<DSC_CUDA_BLOCKS(n),
                                                 DSC_CUDA_DEFAULT_THREADS>>>(xa_data,
                                                                             xb_data,
                                                                             out_data,
                                                                             n,
                                                                             op);
    } else if (xb_scalar) {
        k_binary_op<T, Op, false, true, false><<<DSC_CUDA_BLOCKS(n),
                                                 DSC_CUDA_DEFAULT_THREADS>>>(xa_data,
                                                                             xb_data,
                                                                             out_data,
                                                                             n,
                                                                             op);
    } else if (same_shape) {
        k_binary_op<T, Op, false, false, true><<<DSC_CUDA_BLOCKS(n),
                                                 DSC_CUDA_DEFAULT_THREADS>>>(xa_data,
                                                                             xb_data,
                                                                             out_data,
                                                                             n,
                                                                             op);
    } else {
        binary_params params{.n_dim = out->n_dim};
        memcpy(params.out_shape, out->shape, DSC_MAX_DIMS * sizeof(*out->shape));
        memcpy(params.xa_stride, xa->stride, DSC_MAX_DIMS * sizeof(*xa->stride));
        memcpy(params.xb_stride, xb->stride, DSC_MAX_DIMS * sizeof(*xb->stride));

        // Set the stride = 0 in the broadcast dim
        for (int i = 0; i < DSC_MAX_DIMS; ++i) {
            if (out->shape[i] != 1) {
                if (xa->shape[i] == 1) params.xa_stride[i] = 0;
                if (xb->shape[i] == 1) params.xb_stride[i] = 0;
            }
        }

        k_binary_op<T, Op, false, false, false><<<DSC_CUDA_BLOCKS(n),
                                                  DSC_CUDA_DEFAULT_THREADS>>>(xa_data,
                                                                              xb_data,
                                                                              out_data,
                                                                              n,
                                                                              op,
                                                                              params);
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

void dsc_cuda_add(dsc_device *,
                  const dsc_tensor *xa,
                  const dsc_tensor *xb,
                  dsc_tensor *out) {
    binary_op(xa, xb, out, cuda_add_op());
}

void dsc_cuda_sub(dsc_device *,
                  const dsc_tensor *xa,
                  const dsc_tensor *xb,
                  dsc_tensor *out) {
    binary_op(xa, xb, out, cuda_sub_op());
}

void dsc_cuda_mul(dsc_device *,
                  const dsc_tensor *xa,
                  const dsc_tensor *xb,
                  dsc_tensor *out) {
    binary_op(xa, xb, out, cuda_mul_op());
}

void dsc_cuda_div(dsc_device *,
                  const dsc_tensor *xa,
                  const dsc_tensor *xb,
                  dsc_tensor *out) {
    binary_op(xa, xb, out, cuda_div_op());
}

void dsc_cuda_pow(dsc_device *,
                  const dsc_tensor *xa,
                  const dsc_tensor *xb,
                  dsc_tensor *out) {
    binary_op(xa, xb, out, cuda_pow_op());
}

// ============================================================
// Unary Operations

template<typename Tx, typename To = Tx, typename Op>
static DSC_CUDA_KERNEL void k_unary_op(const Tx *DSC_RESTRICT x,
                                       To *DSC_RESTRICT out,
                                       const int n, Op op) {
    DSC_CUDA_TID();
    DSC_CUDA_STRIDE();

    for (int i = tid; i < n; i += stride) {
        out[i] = op(x[i]);
    }
}

template<typename Op>
static DSC_INLINE void unary_op(const dsc_tensor *DSC_RESTRICT x,
                                dsc_tensor *DSC_RESTRICT out, Op op) {
    const int n = x->ne;
    switch (x->dtype) {
        case F32: {
            DSC_TENSOR_DATA(f32, x);
            DSC_TENSOR_DATA(f32, out);
            k_unary_op<f32><<<DSC_CUDA_BLOCKS(n),
                              DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, op);
            break;
        }
        case F64: {
            DSC_TENSOR_DATA(f64, x);
            DSC_TENSOR_DATA(f64, out);
            k_unary_op<f64><<<DSC_CUDA_BLOCKS(n),
                              DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, op);
            break;
        }
        case C32: {
            DSC_TENSOR_DATA(c32, x);
            DSC_TENSOR_DATA(c32, out);
            k_unary_op<c32><<<DSC_CUDA_BLOCKS(n),
                              DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, op);
            break;
        }
        case C64: {
            DSC_TENSOR_DATA(c64, x);
            DSC_TENSOR_DATA(c64, out);
            k_unary_op<c64><<<DSC_CUDA_BLOCKS(n),
                              DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, op);
            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

void dsc_cuda_cos(dsc_device *,
                 const dsc_tensor *DSC_RESTRICT x,
                 dsc_tensor *DSC_RESTRICT out) {
    unary_op(x, out, cuda_cos_op());
}

void dsc_cuda_sin(dsc_device *,
                 const dsc_tensor *DSC_RESTRICT x,
                 dsc_tensor *DSC_RESTRICT out) {
    unary_op(x, out, cuda_sin_op());
}

void dsc_cuda_sinc(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out) {
    unary_op(x, out, cuda_sinc_op());
}

void dsc_cuda_logn(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out) {
    unary_op(x, out, cuda_logn_op());
}

void dsc_cuda_log2(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out) {
    unary_op(x, out, cuda_log2_op());
}

void dsc_cuda_log10(dsc_device *,
                   const dsc_tensor *DSC_RESTRICT x,
                   dsc_tensor *DSC_RESTRICT out) {
    unary_op(x, out, cuda_log10_op());
}

void dsc_cuda_exp(dsc_device *,
                 const dsc_tensor *DSC_RESTRICT x,
                 dsc_tensor *DSC_RESTRICT out) {
    unary_op(x, out, cuda_exp_op());
}

void dsc_cuda_sqrt(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out) {
    unary_op(x, out, cuda_sqrt_op());
}

void dsc_cuda_i0(dsc_device *,
                 const dsc_tensor *DSC_RESTRICT x,
                 dsc_tensor *DSC_RESTRICT out) {
    const int n = x->ne;
    switch (x->dtype) {
        case F32: {
            // TODO: it's probably a good idea to add a templated flag in order to enable/disable
            //  the generation of float/complex instantiations (also on the CPU side!) This way I don't
            //  have to write the same function n-times.
            //  Also, if the way we launch kernels is good enough (ie. no fancy tricks to control the launch
            //  parameters) it's probably worth using a macro to make it more concise.
            DSC_TENSOR_DATA(f32, x);
            DSC_TENSOR_DATA(f32, out);
            k_unary_op<f32><<<DSC_CUDA_BLOCKS(n),
                              DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_i0_op());
            break;
        }
        case F64: {
            DSC_TENSOR_DATA(f64, x);
            DSC_TENSOR_DATA(f64, out);
            k_unary_op<f64><<<DSC_CUDA_BLOCKS(n),
                              DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_i0_op());
            break;
        }
        DSC_INVALID_CASE("dtype must be real");
    }
}

void dsc_cuda_abs(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out) {
    const int n = x->ne;
    switch (x->dtype) {
        case F32: {
            DSC_TENSOR_DATA(f32, x);
            DSC_TENSOR_DATA(f32, out);
            k_unary_op<f32, f32><<<DSC_CUDA_BLOCKS(n),
                                   DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_abs_op());
            break;
        }
        case F64: {
            DSC_TENSOR_DATA(f64, x);
            DSC_TENSOR_DATA(f64, out);
            k_unary_op<f64, f64><<<DSC_CUDA_BLOCKS(n),
                                   DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_abs_op());
            break;
        }
        case C32: {
            DSC_TENSOR_DATA(c32, x);
            DSC_TENSOR_DATA(f32, out);
            k_unary_op<c32, f32><<<DSC_CUDA_BLOCKS(n),
                                   DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_abs_op());
            break;
        }
        case C64: {
            DSC_TENSOR_DATA(c64, x);
            DSC_TENSOR_DATA(f64, out);
            k_unary_op<c64, f64><<<DSC_CUDA_BLOCKS(n),
                                   DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_abs_op());
            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

void dsc_cuda_angle(dsc_device *,
                    const dsc_tensor *DSC_RESTRICT x,
                    dsc_tensor *DSC_RESTRICT out) {
    const int n = x->ne;
    switch (x->dtype) {
        case F32: {
            DSC_TENSOR_DATA(f32, x);
            DSC_TENSOR_DATA(f32, out);
            k_unary_op<f32, f32><<<DSC_CUDA_BLOCKS(n),
                                   DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_atan2_op());
            break;
        }
        case F64: {
            DSC_TENSOR_DATA(f64, x);
            DSC_TENSOR_DATA(f64, out);
            k_unary_op<f64, f64><<<DSC_CUDA_BLOCKS(n),
                                   DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_atan2_op());
            break;
        }
        case C32: {
            DSC_TENSOR_DATA(c32, x);
            DSC_TENSOR_DATA(f32, out);
            k_unary_op<c32, f32><<<DSC_CUDA_BLOCKS(n),
                                   DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_atan2_op());
            break;
        }
        case C64: {
            DSC_TENSOR_DATA(c64, x);
            DSC_TENSOR_DATA(f64, out);
            k_unary_op<c64, f64><<<DSC_CUDA_BLOCKS(n),
                                   DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_atan2_op());
            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

void dsc_cuda_conj(dsc_device *,
                   const dsc_tensor *DSC_RESTRICT x,
                   dsc_tensor *DSC_RESTRICT out) {
    const int n = x->ne;
    switch (x->dtype) {
        case C32: {
            DSC_TENSOR_DATA(c32, x);
            DSC_TENSOR_DATA(c32, out);
            k_unary_op<c32><<<DSC_CUDA_BLOCKS(n),
                              DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_conj_op());
            break;
        }
        case C64: {
            DSC_TENSOR_DATA(c64, x);
            DSC_TENSOR_DATA(c64, out);
            k_unary_op<c64><<<DSC_CUDA_BLOCKS(n),
                              DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_conj_op());
            break;
        }
        DSC_INVALID_CASE("dtype must be complex");
    }
}

void dsc_cuda_real(dsc_device *,
                   const dsc_tensor *DSC_RESTRICT x,
                   dsc_tensor *DSC_RESTRICT out) {
    const int n = x->ne;
    switch (x->dtype) {
        case C32: {
            DSC_TENSOR_DATA(c32, x);
            DSC_TENSOR_DATA(f32, out);
            k_unary_op<c32, f32><<<DSC_CUDA_BLOCKS(n),
                                   DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_real_op());
            break;
        }
        case C64: {
            DSC_TENSOR_DATA(c64, x);
            DSC_TENSOR_DATA(f64, out);
            k_unary_op<c64, f64><<<DSC_CUDA_BLOCKS(n),
                                   DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_real_op());
            break;
        }
        DSC_INVALID_CASE("dtype must be complex");
    }
}

void dsc_cuda_imag(dsc_device *,
                   const dsc_tensor *DSC_RESTRICT x,
                   dsc_tensor *DSC_RESTRICT out) {
    const int n = x->ne;
    switch (x->dtype) {
        case F32: {
            DSC_TENSOR_DATA(f32, x);
            DSC_TENSOR_DATA(f32, out);
            k_unary_op<f32, f32><<<DSC_CUDA_BLOCKS(n),
                                   DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_imag_op());
            break;
        }
        case F64: {
            DSC_TENSOR_DATA(f64, x);
            DSC_TENSOR_DATA(f64, out);
            k_unary_op<f64, f64><<<DSC_CUDA_BLOCKS(n),
                                   DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_imag_op());
            break;
        }
        case C32: {
            DSC_TENSOR_DATA(c32, x);
            DSC_TENSOR_DATA(f32, out);
            k_unary_op<c32, f32><<<DSC_CUDA_BLOCKS(n),
                                   DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_imag_op());
            break;
        }
        case C64: {
            DSC_TENSOR_DATA(c64, x);
            DSC_TENSOR_DATA(f64, out);
            k_unary_op<c64, f64><<<DSC_CUDA_BLOCKS(n),
                                   DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_imag_op());
            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

void dsc_cuda_clip(dsc_device *,
                   const dsc_tensor *DSC_RESTRICT x,
                   dsc_tensor *DSC_RESTRICT out,
                   const f64 x_min, const f64 x_max) {
    const int n = x->ne;
    switch (out->dtype) {
        case F32: {
            DSC_TENSOR_DATA(f32, x);
            DSC_TENSOR_DATA(f32, out);
            k_unary_op<f32><<<DSC_CUDA_BLOCKS(n),
                              DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_clip_op((f32) x_min, (f32) x_max));
            break;
        }
        case F64: {
            DSC_TENSOR_DATA(f64, x);
            DSC_TENSOR_DATA(f64, out);
            k_unary_op<f64><<<DSC_CUDA_BLOCKS(n),
                              DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_clip_op(x_min, x_max));
            break;
        }
        case C32: {
            DSC_TENSOR_DATA(c32, x);
            DSC_TENSOR_DATA(c32, out);
            k_unary_op<c32><<<DSC_CUDA_BLOCKS(n),
                              DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n,
                                                          cuda_clip_op(dsc_complex(c32, (f32) x_min, dsc_zero<f32>()),
                                                                       dsc_complex(c32, (f32) x_max, dsc_zero<f32>())));
            break;
        }
        case C64: {
            DSC_TENSOR_DATA(c64, x);
            DSC_TENSOR_DATA(c64, out);
            k_unary_op<c64><<<DSC_CUDA_BLOCKS(n),
                              DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n,
                                                          cuda_clip_op(dsc_complex(c64, x_min, dsc_zero<f64>()),
                                                                       dsc_complex(c64, x_max, dsc_zero<f64>())));
            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }
}

// ============================================================
// Unary Operations Along Axis

struct reduce_params {
    int x_shape[DSC_MAX_DIMS]{}, x_stride[DSC_MAX_DIMS]{};
    int out_shape[DSC_MAX_DIMS]{}, out_stride[DSC_MAX_DIMS]{};
    int axis_idx{}, x_ne{}, out_ne{};
};

template <typename T>
static DSC_CUDA_KERNEL void k_sum(const T *DSC_RESTRICT x,
                                  T *DSC_RESTRICT out,
                                  const reduce_params params) {
    DSC_CUDA_TID();

    if (tid >= params.x_ne) return;

    __shared__ T x_s[DSC_CUDA_DEFAULT_THREADS];

    const int axis_n = params.x_shape[params.axis_idx];

    int idx_arr[DSC_MAX_DIMS];
    compute_idx_from_linear(idx_arr, tid, params.out_shape);

    T acc = dsc_zero<T>();

    for (int i = (int) threadIdx.x; i < axis_n; i += (int) blockDim.x) {
        idx_arr[params.axis_idx] = i;
        const int idx = compute_linear_idx(idx_arr, params.x_stride);
        const T val = x[idx];
        acc = cuda_add_op()(acc, val);
    }
    x_s[threadIdx.x] = acc;
    __syncthreads();

    for (int i = (int) (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            x_s[threadIdx.x] = cuda_add_op()(x_s[threadIdx.x], x_s[threadIdx.x + i]);
        }
        __syncthreads();
    }

    if (tid < params.out_ne) {
        // Reset the idx array
        idx_arr[params.axis_idx] = 0;
        const int idx = compute_linear_idx(idx_arr, params.out_stride);
        cuda_atomic_add_op()(&out[idx], *x_s);
    }
}

template <typename T>
static DSC_INLINE void sum(const dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out,
                           const int axis_idx) {
    DSC_TENSOR_DATA_R(T, x);
    DSC_TENSOR_DATA_R(T, out);

    cudaMemset(out_data, 0, out->ne * DSC_DTYPE_SIZE[out->dtype]);

    reduce_params params{.axis_idx = axis_idx, .x_ne = x->ne, .out_ne = out->ne};
    memcpy(params.x_shape, x->shape, DSC_MAX_DIMS * sizeof(*x->shape));
    memcpy(params.x_stride, x->stride, DSC_MAX_DIMS * sizeof(*x->stride));
    memcpy(params.out_shape, out->shape, DSC_MAX_DIMS * sizeof(*out->shape));
    memcpy(params.out_stride, out->stride, DSC_MAX_DIMS * sizeof(*out->stride));

    k_sum<T><<<DSC_CUDA_BLOCKS(out->ne), DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, params);
}

void dsc_cuda_sum(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out,
                  const int axis_idx) {
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

