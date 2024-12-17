// Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#include "cuda/dsc_cuda.h"
#include "cuda/dsc_ops.h"
#include "dsc_device.h"

template<typename Tx, typename To>
static DSC_CUDA_KERNEL void k_cast_op(const Tx *DSC_RESTRICT x,
                                      To *DSC_RESTRICT out, const int n) {
    DSC_CUDA_TID();
    DSC_CUDA_STRIDE();

    for (size i = tid; i < n; i += stride) {
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

    for (size i = tid; i < n; i += stride) {
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

    for (size i = tid; i < n; i += stride) {
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
// Binary Operations

template<typename T, typename Op>
static DSC_CUDA_KERNEL void binary_op(const dsc_tensor *xa,
                                      const dsc_tensor *xb,
                                      dsc_tensor *out,
                                      Op op) {
    // T *xa_data = (T *) xa->data;
    // T *xb_data = (T *) xb->data;
    // T *out_data = (T *) out->data;
    // const bool xa_scalar = xa->n_dim == 1 && xa->shape[dsc_tensor_dim(xa, -1)] == 1;
    // const bool xb_scalar = xb->n_dim == 1 && xb->shape[dsc_tensor_dim(xb, -1)] == 1;
    //
    // if (xa_scalar) {
    //     const T val = xa_data[0];
    //     dsc_for(i, out) {
    //         out_data[i] = op(
    //                 val,
    //                 xb_data[i]
    //         );
    //     }
    // } else if (xb_scalar) {
    //     const T val = xb_data[0];
    //     dsc_for(i, out) {
    //         out_data[i] = op(
    //                 xa_data[i],
    //                 val
    //         );
    //     }
    // } else {
    //     dsc_broadcast_iterator xa_it(xa, out->shape), xb_it(xb, out->shape);
    //     dsc_for(i, out) {
    //         out_data[i] = op(
    //                 xa_data[xa_it.index()],
    //                 xb_data[xb_it.index()]
    //         );
    //         xa_it.next(), xb_it.next();
    //     }
    // }
}

template<typename Op>
static DSC_INLINE void binary_op(const dsc_tensor *xa,
                                 const dsc_tensor *xb,
                                 dsc_tensor *out,
                                 Op op) {
    // switch (out->dtype) {
    //     case F32:
    //         binary_op<f32>(xa, xb, out, op);
    //         break;
    //     case F64:
    //         binary_op<f64>(xa, xb, out, op);
    //         break;
    //     case C32:
    //         binary_op<c32>(xa, xb, out, op);
    //         break;
    //     case C64:
    //         binary_op<c64>(xa, xb, out, op);
    //         break;
    //     DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    // }
}

void dsc_cuda_add(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT xa,
                  const dsc_tensor *DSC_RESTRICT xb,
                  dsc_tensor *DSC_RESTRICT out) {
}

void dsc_cuda_sub(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT xa,
                  const dsc_tensor *DSC_RESTRICT xb,
                  dsc_tensor *DSC_RESTRICT out) {
}

void dsc_cuda_mul(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT xa,
                  const dsc_tensor *DSC_RESTRICT xb,
                  dsc_tensor *DSC_RESTRICT out) {
}

void dsc_cuda_div(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT xa,
                  const dsc_tensor *DSC_RESTRICT xb,
                  dsc_tensor *DSC_RESTRICT out) {
}

void dsc_cuda_pow(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT xa,
                  const dsc_tensor *DSC_RESTRICT xb,
                  dsc_tensor *DSC_RESTRICT out) {
}

// ============================================================
// Unary Operations

template<typename Tx, typename To = Tx, typename Op>
static DSC_CUDA_KERNEL void k_unary_op(const Tx *DSC_RESTRICT x,
                                       To *DSC_RESTRICT out,
                                       const int n, Op op) {
    DSC_CUDA_TID();
    DSC_CUDA_STRIDE();

    for (size i = tid; i < n; i += stride) {
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
            k_unary_op<f32, f32>
                    <<<DSC_CUDA_BLOCKS(n), DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_abs_op());
            break;
        }
        case F64: {
            DSC_TENSOR_DATA(f64, x);
            DSC_TENSOR_DATA(f64, out);
            k_unary_op<f64, f64>
                    <<<DSC_CUDA_BLOCKS(n), DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_abs_op());
            break;
        }
        case C32: {
            DSC_TENSOR_DATA(c32, x);
            DSC_TENSOR_DATA(f32, out);
            k_unary_op<c32, f32>
                    <<<DSC_CUDA_BLOCKS(n), DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_abs_op());
            break;
        }
        case C64: {
            DSC_TENSOR_DATA(c64, x);
            DSC_TENSOR_DATA(f64, out);
            k_unary_op<c64, f64>
                    <<<DSC_CUDA_BLOCKS(n), DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_abs_op());
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
            k_unary_op<f32, f32>
                    <<<DSC_CUDA_BLOCKS(n), DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_atan2_op());
            break;
        }
        case F64: {
            DSC_TENSOR_DATA(f64, x);
            DSC_TENSOR_DATA(f64, out);
            k_unary_op<f64, f64>
                    <<<DSC_CUDA_BLOCKS(n), DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_atan2_op());
            break;
        }
        case C32: {
            DSC_TENSOR_DATA(c32, x);
            DSC_TENSOR_DATA(f32, out);
            k_unary_op<c32, f32>
                    <<<DSC_CUDA_BLOCKS(n), DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_atan2_op());
            break;
        }
        case C64: {
            DSC_TENSOR_DATA(c64, x);
            DSC_TENSOR_DATA(f64, out);
            k_unary_op<c64, f64>
                    <<<DSC_CUDA_BLOCKS(n), DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_atan2_op());
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
            k_unary_op<c32, f32>
                    <<<DSC_CUDA_BLOCKS(n), DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_real_op());
            break;
        }
        case C64: {
            DSC_TENSOR_DATA(c64, x);
            DSC_TENSOR_DATA(f64, out);
            k_unary_op<c64, f64>
                    <<<DSC_CUDA_BLOCKS(n), DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_real_op());
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
            k_unary_op<f32, f32>
                    <<<DSC_CUDA_BLOCKS(n), DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_imag_op());
            break;
        }
        case F64: {
            DSC_TENSOR_DATA(f64, x);
            DSC_TENSOR_DATA(f64, out);
            k_unary_op<f64, f64>
                    <<<DSC_CUDA_BLOCKS(n), DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_imag_op());
            break;
        }
        case C32: {
            DSC_TENSOR_DATA(c32, x);
            DSC_TENSOR_DATA(f32, out);
            k_unary_op<c32, f32>
                    <<<DSC_CUDA_BLOCKS(n), DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_imag_op());
            break;
        }
        case C64: {
            DSC_TENSOR_DATA(c64, x);
            DSC_TENSOR_DATA(f64, out);
            k_unary_op<c64, f64>
                    <<<DSC_CUDA_BLOCKS(n), DSC_CUDA_DEFAULT_THREADS>>>(x_data, out_data, n, cuda_imag_op());
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
