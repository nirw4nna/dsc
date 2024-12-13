// Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#include "cuda/dsc_cuda.h"
#include "dsc_device.h"

//
// struct cuda_add_op {
//     template<typename T>
//     DSC_INLINE DSC_CUDA_FUNC T operator()(const T xa, const T xb) const noexcept {
//         if constexpr (dsc_is_complex<T>()) {
//             return dsc_complex(T, xa.real + xb.real, xa.imag + xb.imag);
//         } else {
//             return xa + xb;
//         }
//     }
// };
//
// struct cuda_mul_op {
//     template<typename T>
//     DSC_INLINE DSC_CUDA_FUNC T operator()(const T xa, const T xb) const noexcept {
//         if constexpr (dsc_is_complex<T>()) {
//             return dsc_complex(T, (xa.real * xb.real) - (xa.imag * xb.imag),
//                                (xa.real * xb.imag) + (xa.imag * xb.real));
//         } else {
//             return xa * xb;
//         }
//     }
// };

// ============================================================
// Kernels
//

// template <typename T>
// static DSC_CUDA_KERNEL void k_assign_op(T *DSC_RESTRICT x, const int n,
//                                         const T start, const T step) {
//     DSC_CUDA_TID();
//     DSC_CUDA_STRIDE();
//
//     for (size i = tid; i < n; i += stride) {
//         x[i] = cuda_add_op()(start, cuda_mul_op()(step, (T) stride));
//     }
// }

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

// ============================================================
// CUDA-specific operations
//

/*
#define DSC_CUDA_DTYPE_DISPATCH(PTR, func, ...) \
    do {                                    \
        switch ((PTR)->dtype) {             \
            case F32: {                     \
                DSC_TENSOR_DATA(f32, PTR);  \
                func<f32><<<DSC_CUDA_BLOCKS((PTR)->ne), DSC_CUDA_THREADS((PTR)->ne)>>>(##__VA_ARGS__); \
                break;                      \
            }                               \
            case F64: {                     \
                DSC_TENSOR_DATA(f64, PTR);  \
                func<f64><<<DSC_CUDA_BLOCKS((PTR)->ne), DSC_CUDA_THREADS((PTR)->ne)>>>(##__VA_ARGS__); \
                break;                      \
            }                               \
            case C32: {                     \
                DSC_TENSOR_DATA(c32, PTR);  \
                func<c32><<<DSC_CUDA_BLOCKS((PTR)->ne), DSC_CUDA_THREADS((PTR)->ne)>>>(##__VA_ARGS__); \
                break;                      \
            }                               \
            case C64: {                     \
                DSC_TENSOR_DATA(c64, PTR);  \
                func<c64><<<DSC_CUDA_BLOCKS((PTR)->ne), DSC_CUDA_THREADS((PTR)->ne)>>>(##__VA_ARGS__); \
                break;                                          \
            }                                                   \
            DSC_INVALID_CASE("unknown dtype=%d", (PTR)->dtype); \
        }                                                       \
    } while (0)
*/

// Fixme: the way to compute blocks is wrong, should depend on the number of threads
// void dsc_cuda_arange(dsc_device *, dsc_tensor *DSC_RESTRICT x) {
//     switch (x->dtype) {
//         case F32: {
//             DSC_TENSOR_DATA(f32, x);
//             k_assign_op<f32><<<DSC_CUDA_BLOCKS(x->ne),
//                                DSC_CUDA_THREADS(x->ne)>>>(x_data, x->ne, 0.f, (f32) x->ne);
//             break;
//         }
//         case F64: {
//             DSC_TENSOR_DATA(f64, x);
//             k_assign_op<f64><<<DSC_CUDA_BLOCKS(x->ne),
//                                DSC_CUDA_THREADS(x->ne)>>>(x_data, x->ne, 0., (f64) x->ne);
//             break;
//         }
//         case C32: {
//             DSC_TENSOR_DATA(c32, x);
//             k_assign_op<c32><<<DSC_CUDA_BLOCKS(x->ne),
//                                DSC_CUDA_THREADS(x->ne)>>>(x_data,
//                                                           x->ne,
//                                                           dsc_complex(c32, 0.f, 0.f),
//                                                           dsc_complex(c32, (f32) x->ne, 0.f));
//             break;
//         }
//         case C64: {
//             DSC_TENSOR_DATA(c64, x);
//             k_assign_op<c64><<<DSC_CUDA_BLOCKS(x->ne),
//                                DSC_CUDA_THREADS(x->ne)>>>(x_data,
//                                                           x->ne,
//                                                           dsc_complex(c64, 0., 0.),
//                                                           dsc_complex(c64, (f64) x->ne, 0.));
//             break;
//         }
//         DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
//     }
// }

void dsc_cuda_randn(dsc_device *dev, dsc_tensor *DSC_RESTRICT x) {
    dsc_cuda_dev_info *info = (dsc_cuda_dev_info *) dev->extra_info;

    switch (x->dtype) {
        case F32: {
            DSC_TENSOR_DATA(f32, x);
            k_randn<f32><<<1, DSC_CUDA_THREADS(x->ne)>>>(info->randState, x_data, x->ne);
            break;
        }
        case F64: {
            DSC_TENSOR_DATA(f64, x);
            k_randn<f64><<<1, DSC_CUDA_THREADS(x->ne)>>>(info->randState, x_data, x->ne);
            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}
