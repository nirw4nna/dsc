// Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#include "cpu/dsc_cpu.h"
#include "dsc_device.h"
#include "dsc_ops.h"
#include <random>

#define dsc_for(idx, X) for (int idx = 0; idx < (X)->ne; ++idx)

// ============================================================
// Ops
//

template<typename T>
static DSC_INLINE void assign_op(dsc_tensor *DSC_RESTRICT x,
                                 const T start, const T step) {
    DSC_TENSOR_DATA(T, x);

    T val = start;
    dsc_for(i, x) {
        x_data[i] = val;
        val = add_op()(val, step);
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

// ============================================================
// CPU-specific operations
//

// void dsc_cpu_arange(dsc_device *, dsc_tensor *DSC_RESTRICT x) {
//     switch (x->dtype) {
//         case F32:
//             assign_op<f32>(x, 0.f, (f32) x->ne);
//             break;
//         case F64:
//             assign_op<f64>(x, 0, (f64) x->ne);
//             break;
//         case C32:
//             assign_op<c32>(x,
//                            dsc_complex(c32, 0.f, 0.f),
//                            dsc_complex(c32, (f32) x->ne, 0.f));
//             break;
//         case C64:
//             assign_op<c64>(x,
//                            dsc_complex(c64, 0., 0.),
//                            dsc_complex(c64, (f64) x->ne, 0.));
//             break;
//         DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
//     }
// }

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