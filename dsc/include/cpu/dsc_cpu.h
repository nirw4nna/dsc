// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "dsc.h"


struct dsc_device;


// ============================================================
// CPU-specific operations
//

extern void dsc_cpu_cast(dsc_device *, const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out);

extern void dsc_cpu_arange(dsc_device *, dsc_tensor *DSC_RESTRICT x);

extern void dsc_cpu_randn(dsc_device *, dsc_tensor *DSC_RESTRICT x);

// ============================================================
// Indexing and Slicing
//

extern void dsc_cpu_get_slice(dsc_device *,
                              const dsc_tensor *DSC_RESTRICT x,
                              dsc_tensor *DSC_RESTRICT out,
                              int n_slices, const dsc_slice *slices,
                              bool whole);

extern void dsc_cpu_set_slice(dsc_device *,
                              dsc_tensor *DSC_RESTRICT xa,
                              bool xa_scalar,
                              const dsc_tensor *DSC_RESTRICT xb,
                              bool xb_scalar,
                              int n_slices,
                              const dsc_slice *slices,
                              bool whole);

// ============================================================
// Binary Operations

extern void dsc_cpu_add(dsc_device *,
                        const dsc_tensor *xa,
                        const dsc_tensor *xb,
                        dsc_tensor *out);

extern void dsc_cpu_sub(dsc_device *,
                        const dsc_tensor *xa,
                        const dsc_tensor *xb,
                        dsc_tensor *out);

extern void dsc_cpu_mul(dsc_device *,
                        const dsc_tensor *xa,
                        const dsc_tensor *xb,
                        dsc_tensor *out);

extern void dsc_cpu_div(dsc_device *,
                        const dsc_tensor *xa,
                        const dsc_tensor *xb,
                        dsc_tensor *out);

extern void dsc_cpu_pow(dsc_device *,
                        const dsc_tensor *xa,
                        const dsc_tensor *xb,
                        dsc_tensor *out);

// ============================================================
// Unary Operations

extern void dsc_cpu_cos(dsc_device *,
                        const dsc_tensor *DSC_RESTRICT x,
                        dsc_tensor *DSC_RESTRICT out);

extern void dsc_cpu_sin(dsc_device *,
                        const dsc_tensor *DSC_RESTRICT x,
                        dsc_tensor *DSC_RESTRICT out);

extern void dsc_cpu_sinc(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out);

extern void dsc_cpu_logn(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out);

extern void dsc_cpu_log2(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out);

extern void dsc_cpu_log10(dsc_device *,
                          const dsc_tensor *DSC_RESTRICT x,
                          dsc_tensor *DSC_RESTRICT out);

extern void dsc_cpu_exp(dsc_device *,
                        const dsc_tensor *DSC_RESTRICT x,
                        dsc_tensor *DSC_RESTRICT out);

extern void dsc_cpu_sqrt(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out);

extern void dsc_cpu_i0(dsc_device *,
                       const dsc_tensor *DSC_RESTRICT x,
                       dsc_tensor *DSC_RESTRICT out);

extern void dsc_cpu_abs(dsc_device *,
                        const dsc_tensor *DSC_RESTRICT x,
                        dsc_tensor *DSC_RESTRICT out);

extern void dsc_cpu_angle(dsc_device *,
                          const dsc_tensor *DSC_RESTRICT x,
                          dsc_tensor *DSC_RESTRICT out);

extern void dsc_cpu_conj(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out);

extern void dsc_cpu_real(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out);

extern void dsc_cpu_imag(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out);

extern void dsc_cpu_clip(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out,
                         f64 x_min, f64 x_max);

// ============================================================
// Unary Operations Along Axis

extern void dsc_cpu_sum(dsc_device *,
                        const dsc_tensor *DSC_RESTRICT x,
                        dsc_tensor *DSC_RESTRICT out,
                        int axis_idx);

extern void dsc_cpu_min(dsc_device *,
                        const dsc_tensor *DSC_RESTRICT x,
                        dsc_tensor *DSC_RESTRICT out,
                        int axis_idx);

extern void dsc_cpu_max(dsc_device *,
                        const dsc_tensor *DSC_RESTRICT x,
                        dsc_tensor *DSC_RESTRICT out,
                        int axis_idx);