// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#include "cuda/dsc_cuda.h"
#include "cuda/dsc_iter.h"
#include "cuda/dsc_ops.h"
#include "dsc_device.h"

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
    int n_dim, n_slices;
};

template<typename T>
static DSC_CUDA_KERNEL void k_get_slice(const T *DSC_RESTRICT x,
                                        T *DSC_RESTRICT out,
                                        const int n,
                                        slicing_params params) {
    DSC_CUDA_TID();
    DSC_CUDA_STRIDE();

    if (tid >= n) return;

    dsc_slice_iterator it(params.shape, params.stride,
                            params.n_dim, params.n_slices,
                            params.slices);

    it.advance(tid);

    if (!it.has_next()) return;
    
    for (int i = tid; i < n && it.has_next(); i += stride) {
        out[i] = x[it.index()];
        it.advance(stride);
    }
}

void dsc_cuda_get_slice(dsc_device *,
                        const dsc_tensor *DSC_RESTRICT x,
                        dsc_tensor *DSC_RESTRICT out,
                        const int n_slices, const dsc_slice *slices) {
    slicing_params params{.n_dim = x->n_dim, .n_slices = n_slices};
    memcpy(params.shape, x->shape, DSC_MAX_DIMS * sizeof(*x->shape));
    memcpy(params.stride, x->stride, DSC_MAX_DIMS * sizeof(*x->stride));
    memcpy(params.slices, slices, n_slices * sizeof(*slices));

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
                                        const int n, slicing_params params) {
    DSC_CUDA_TID();
    DSC_CUDA_STRIDE();

    if (tid >= n) return;

    dsc_slice_iterator it(params.shape, params.stride,
                          params.n_dim, params.n_slices,
                          params.slices);

    it.advance(tid);

    if (!it.has_next()) return;

    for (int i = tid; i < n && it.has_next(); i += stride) {
        if (xb_scalar) {
            // TODO: (1)
            xa[it.index()] = xb[0];
        } else {
            xa[it.index()] = xb[i];
        }
        it.advance(stride);
    }
}

void dsc_cuda_set_slice(dsc_device *dev,
                        dsc_tensor *DSC_RESTRICT xa,
                        const bool xa_scalar,
                        const dsc_tensor *DSC_RESTRICT xb,
                        const bool xb_scalar,
                        const int n_slices, const dsc_slice *slices) {
    if (xa_scalar) {
        int offset = 0;
        for (int i = 0; i < n_slices; ++i)
            offset += (slices[i].start * xa->stride[dsc_tensor_dim(xa, i)]);

        DSC_TENSOR_DATA_R(byte, xa);
        DSC_TENSOR_DATA_R(void, xb);
        dev->memcpy(xa_data + (offset * DSC_DTYPE_SIZE[xa->dtype]),
                    xb_data, DSC_DTYPE_SIZE[xa->dtype], ON_DEVICE);
    } else {
        slicing_params params{.n_dim = xa->n_dim, .n_slices = n_slices};
        memcpy(params.shape, xa->shape, DSC_MAX_DIMS * sizeof(*xa->shape));
        memcpy(params.stride, xa->stride, DSC_MAX_DIMS * sizeof(*xa->stride));
        memcpy(params.slices, slices, n_slices * sizeof(*slices));

        const int n = xa->ne;
        switch (xa->dtype) {
            case F32: {
                DSC_TENSOR_DATA_R(f32, xa);
                DSC_TENSOR_DATA_R(f32, xb);
                k_set_slice<f32><<<DSC_CUDA_BLOCKS(n),
                                   DSC_CUDA_DEFAULT_THREADS>>>(xa_data, xb_data, xb_scalar, n, params);
                break;
            }
            case F64: {
                DSC_TENSOR_DATA_R(f64, xa);
                DSC_TENSOR_DATA_R(f64, xb);
                k_set_slice<f64><<<DSC_CUDA_BLOCKS(n),
                                   DSC_CUDA_DEFAULT_THREADS>>>(xa_data, xb_data, xb_scalar, n, params);
                break;
            }
            case C32: {
                DSC_TENSOR_DATA_R(c32, xa);
                DSC_TENSOR_DATA_R(c32, xb);
                k_set_slice<c32><<<DSC_CUDA_BLOCKS(n),
                                   DSC_CUDA_DEFAULT_THREADS>>>(xa_data, xb_data, xb_scalar, n, params);
                break;
            }
            case C64: {
                DSC_TENSOR_DATA_R(c64, xa);
                DSC_TENSOR_DATA_R(c64, xb);
                k_set_slice<c64><<<DSC_CUDA_BLOCKS(n),
                                   DSC_CUDA_DEFAULT_THREADS>>>(xa_data, xb_data, xb_scalar, n, params);
                break;
            }
            DSC_INVALID_CASE("unknown dtype=%d", xa->dtype);
        }
    }
}

// ============================================================
// Binary Operations

struct binary_params {
    int out_shape[DSC_MAX_DIMS];
    int xa_stride[DSC_MAX_DIMS], xb_stride[DSC_MAX_DIMS];
};

template<int N, int Cur = 0>
static DSC_CUDA_FUNC DSC_INLINE void unroll_index(const int i, int *unrolled,
                                                  const int *shape, const int prod = 1) {
    if constexpr (Cur == N - 1) {
        unrolled[Cur] = i / prod;
    } else {
        unrolled[Cur] = (i / prod) % shape[Cur];
        unroll_index<N, Cur + 1>(i, unrolled, shape, prod * shape[Cur]);
    }
}

template<int N, int Cur = 0>
static DSC_CUDA_FUNC DSC_INLINE int compute_index(const int *unrolled_i, const int *stride) {
    if constexpr (Cur == N) {
        return 0;
    } else {
        return unrolled_i[Cur] * stride[Cur] + compute_index<N, Cur + 1>(unrolled_i, stride);
    }
}

template<typename T, typename Op,
         bool xa_scalar = false,
         bool xb_scalar = false,
         bool shape_matches = false>
static DSC_CUDA_KERNEL void k_binary_op(const T *xa, const T *xb, T *out,
                                        const int n, Op op,
                                        const binary_params params = {}) {
    DSC_CUDA_TID();
    DSC_CUDA_STRIDE();

    for (int i = tid; i < n; i += stride) {
        if constexpr (xa_scalar) {
            out[i] = op(xa[0], xb[i]);
        } else if constexpr (xb_scalar) {
            out[i] = op(xa[i], xb[0]);
        } else if constexpr (shape_matches) {
            out[i] = op(xa[i], xb[i]);
        } else {
            // In this case we need to apply broadcasting
            int unrolled_i[DSC_MAX_DIMS];

            unroll_index<DSC_MAX_DIMS>((int) i, unrolled_i, params.out_shape);
            const int xa_idx = compute_index<DSC_MAX_DIMS>(unrolled_i, params.xa_stride);
            const int xb_idx = compute_index<DSC_MAX_DIMS>(unrolled_i, params.xb_stride);

            out[i] = op(xa[xa_idx], xb[xb_idx]);
        }
    }
}

template<typename T, typename Op>
static DSC_INLINE void binary_op(const dsc_tensor *xa,
                                 const dsc_tensor *xb,
                                 dsc_tensor *out,
                                 Op op) {
    const bool xa_scalar = xa->n_dim == 1 && xa->shape[dsc_tensor_dim(xa, -1)] == 1;
    const bool xb_scalar = xb->n_dim == 1 && xb->shape[dsc_tensor_dim(xb, -1)] == 1;
    const bool same_shape = xa->n_dim == xb->n_dim && memcmp(&xa->shape[dsc_tensor_dim(xa, -1)],
                                                             &xb->shape[dsc_tensor_dim(xb, -1)],
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
        binary_params params{};
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

void dsc_cuda_sum(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out,
                  const int axis_idx) {
    switch (out->dtype) {
        case F32: {
    
            break;
        }
        case F64: {

            break;
        }
        case C32: {

            break;
        }
        case C64: {

            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }
}

