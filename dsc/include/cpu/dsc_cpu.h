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

extern void dsc_cpu_arange(dsc_device *,
                           dsc_tensor *DSC_RESTRICT x,
                           f64 start, f64 step);

extern void dsc_cpu_repeat(dsc_device *,
                           const dsc_tensor *DSC_RESTRICT x,
                           dsc_tensor *DSC_RESTRICT out,
                           int repeats, int axis_idx);

extern void dsc_cpu_randn(dsc_device *, dsc_tensor *DSC_RESTRICT x);

extern void dsc_cpu_topk(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT tmp_values,
                         dsc_tensor *DSC_RESTRICT tmp_indexes,
                         dsc_tensor *DSC_RESTRICT out_values,
                         dsc_tensor *DSC_RESTRICT out_indexes,
                         int k, int axis_idx,
                         bool largest);

extern void dsc_cpu_multinomial(dsc_device *,
                                const dsc_tensor *DSC_RESTRICT x,
                                dsc_tensor *DSC_RESTRICT out,
                                int num_samples);

// ============================================================
// Tensor Manipulation

extern void dsc_cpu_concat(dsc_device *,
                           dsc_tensor **to_concat,
                           int tensors,
                           dsc_tensor *DSC_RESTRICT out,
                           int axis_idx);

extern void dsc_cpu_split(dsc_device *,
                          const dsc_tensor *DSC_RESTRICT x,
                          dsc_tensor *DSC_RESTRICT out,
                          int axis_idx, int ne, int offset);

extern void dsc_cpu_transpose(dsc_device *,
                              const dsc_tensor *DSC_RESTRICT x,
                              dsc_tensor *DSC_RESTRICT out,
                              const int *new_shape,
                              const int *new_stride);

extern void dsc_cpu_tril(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         int diagonal,
                         dsc_tensor *DSC_RESTRICT out);

// ============================================================
// Indexing and Slicing
//

extern void dsc_cpu_get_slice(dsc_device *,
                              const dsc_tensor *DSC_RESTRICT x,
                              dsc_tensor *DSC_RESTRICT out,
                              int n_slices, const dsc_slice *slices,
                              bool whole);

extern void dsc_cpu_get_tensor(dsc_device *,
                               const dsc_tensor *DSC_RESTRICT x,
                               const dsc_tensor *DSC_RESTRICT indexes,
                               dsc_tensor *DSC_RESTRICT out);

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

extern void dsc_cpu_matmul(dsc_device *dev,
                           const dsc_tensor *DSC_RESTRICT xa,
                           const dsc_tensor *DSC_RESTRICT xb,
                           bool trans_b,
                           dsc_tensor *DSC_RESTRICT out);

extern void dsc_cpu_compare(dsc_device *,
                            const dsc_tensor *xa,
                            const dsc_tensor *xb,
                            dsc_comparison_op comp,
                            dsc_tensor *out);

extern void dsc_cpu_masked_fill(dsc_device *,
                                dsc_tensor *x,
                                const dsc_tensor *mask,
                                f64 value);

extern void dsc_cpu_outer(dsc_device *,
                          const dsc_tensor *DSC_RESTRICT xa,
                          const dsc_tensor *DSC_RESTRICT xb,
                          dsc_tensor *DSC_RESTRICT out);

extern void dsc_cpu_where(dsc_device *,
                          const dsc_tensor *DSC_RESTRICT condition,
                          const dsc_tensor *DSC_RESTRICT input,
                          const dsc_tensor *DSC_RESTRICT other,
                          dsc_tensor *DSC_RESTRICT out);

// ============================================================
// Unary Operations

extern void dsc_cpu_cos(dsc_device *,
                        const dsc_tensor *DSC_RESTRICT x,
                        dsc_tensor *DSC_RESTRICT out);

extern void dsc_cpu_sin(dsc_device *,
                        const dsc_tensor *DSC_RESTRICT x,
                        dsc_tensor *DSC_RESTRICT out);

extern void dsc_cpu_tanh(dsc_device *dev,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out);

extern void dsc_cpu_exp(dsc_device *,
                        const dsc_tensor *DSC_RESTRICT x,
                        dsc_tensor *DSC_RESTRICT out);

extern void dsc_cpu_sqrt(dsc_device *,
                         const dsc_tensor *DSC_RESTRICT x,
                         dsc_tensor *DSC_RESTRICT out);

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