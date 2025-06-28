// Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#include "gpu/dsc_gpu.h"
#include "dsc_device.h"
#include "gpu/dsc_ops.h"


#define init_slice_idx(ARR, SLICES) \
    for (int i__ = 0; i__ < DSC_MAX_DIMS; ++i__) ARR[i__] = SLICES[i__].start


// ============================================================
// GPU-specific operations
//

template<typename Tx, typename To>
static DSC_GPU_KERNEL void k_cast_op(const Tx *DSC_RESTRICT x,
                                     To *DSC_RESTRICT out, const int n) {
    DSC_GPU_TID();
    DSC_GPU_STRIDE();

    for (int i = tid; i < n; i += stride) {
        out[i] = gpu_cast_op().operator()<Tx, To>(x[i]);
    }
}

template<typename Tx>
static DSC_INLINE void cast_op(const dsc_tensor *DSC_RESTRICT x,
                               dsc_tensor *DSC_RESTRICT out) {
    DSC_DATA(Tx, x);

    const int n = x->ne;
    switch (out->dtype) {
        case BOOL: {
            DSC_DATA(bool, out);
            k_cast_op<Tx, bool><<<DSC_GPU_BLOCKS(n),
                                  DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, n);
            break;
        }
        case I32: {
            DSC_DATA(i32, out);
            k_cast_op<Tx, i32><<<DSC_GPU_BLOCKS(n),
                                 DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, n);
            break;
        }
        case BF16: {
            DSC_DATA(bf16, out);
            k_cast_op<Tx, bf16><<<DSC_GPU_BLOCKS(n),
                                  DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, n);
            break;
        }
        case F32: {
            DSC_DATA(f32, out);
            k_cast_op<Tx, f32><<<DSC_GPU_BLOCKS(n),
                                 DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, n);
            break;
        }
        case F64: {
            DSC_DATA(f64, out);
            k_cast_op<Tx, f64><<<DSC_GPU_BLOCKS(n),
                                 DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, n);
            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

void dsc_gpu_cast(dsc_device *, const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out) {
    switch (x->dtype) {
        case BOOL:
            cast_op<bool>(x, out);
            break;
        case I32:
            cast_op<i32>(x, out);
            break;
        case BF16:
            cast_op<bf16>(x, out);
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
static DSC_GPU_KERNEL void k_assign_op(T *DSC_RESTRICT x, const int n,
                                       const T start, const T step) {
    DSC_GPU_TID();
    DSC_GPU_STRIDE();

    for (int i = tid; i < n; i += stride) {
        x[i] = gpu_add_op()(start, gpu_mul_op()(step, (T) i));
    }
}

void dsc_gpu_arange(dsc_device *,
                    dsc_tensor *DSC_RESTRICT x,
                    const f64 start, const f64 step) {
    const int n = x->ne;
    switch (x->dtype) {
        case I32: {
            DSC_DATA(i32, x);
            k_assign_op<i32><<<DSC_GPU_BLOCKS(n),
                               DSC_GPU_DEFAULT_THREADS>>>(x_data, n, (i32) start, (i32) step);
            break;
        }
        case BF16: {
            DSC_DATA(bf16, x);
            k_assign_op<bf16><<<DSC_GPU_BLOCKS(n),
                                DSC_GPU_DEFAULT_THREADS>>>(x_data, n, start, step);
            break;
        }
        case F32: {
            DSC_DATA(f32, x);
            k_assign_op<f32><<<DSC_GPU_BLOCKS(n),
                               DSC_GPU_DEFAULT_THREADS>>>(x_data, n, (f32) start, (f32) step);
            break;
        }
        case F64: {
            DSC_DATA(f64, x);
            k_assign_op<f64><<<DSC_GPU_BLOCKS(n),
                               DSC_GPU_DEFAULT_THREADS>>>(x_data, n, start, step);
            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

static DSC_INLINE DSC_GPU_FUNC int compute_linear_idx(const int *idx_arr,
                                                      const int *stride) {
    int linear_idx = 0;
    for (int i = 0; i < DSC_MAX_DIMS; ++i) {
        linear_idx += idx_arr[i] * stride[i];
    }
    return linear_idx;
}

static DSC_INLINE DSC_GPU_FUNC void compute_idx_from_linear(int *idx_arr, int linear_idx,
                                                            const int *shape) {
    for (int i = DSC_MAX_DIMS - 1; i >= 0; i--) {
        idx_arr[i] = linear_idx % shape[i];
        linear_idx /= shape[i];
    }
}

struct repeat_params {
    int out_shape[DSC_MAX_DIMS]{};
    int x_stride[DSC_MAX_DIMS]{};
};

template<typename T>
static DSC_GPU_KERNEL void k_repeat(const T *DSC_RESTRICT x,
                                    T *DSC_RESTRICT out,
                                    const int n, const int repeats,
                                    const int axis_idx, const repeat_params params) {
    DSC_GPU_TID();
    DSC_GPU_STRIDE();

    if (tid >= n) return;

    int out_idx[DSC_MAX_DIMS];
    for (int i = tid; i < n; i += stride) {
        // Compute the index in the output array. Since we are repeating along a given axis
        // the index in the input array is simply idx[axis] / repeats
        compute_idx_from_linear(out_idx, i, params.out_shape);
        out_idx[axis_idx] /= repeats;
        out[i] = x[compute_linear_idx(out_idx, params.x_stride)];
    }
}

void dsc_gpu_repeat(dsc_device *,
                    const dsc_tensor *DSC_RESTRICT x,
                    dsc_tensor *DSC_RESTRICT out,
                    const int repeats, const int axis_idx) {
    const int n = out->ne;

    // Prepare kernel params
    repeat_params params{};
    memcpy(params.out_shape, out->shape, DSC_MAX_DIMS * sizeof(*out->shape));
    memcpy(params.x_stride, x->stride, DSC_MAX_DIMS * sizeof(*x->stride));

    switch (out->dtype) {
        case BOOL: {
            DSC_DATA(bool, x);
            DSC_DATA(bool, out);
            k_repeat<<<DSC_GPU_BLOCKS(n), DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data,
                                                                     n, repeats,
                                                                     axis_idx, params);
            break;
        }
        case I32: {
            DSC_DATA(i32, x);
            DSC_DATA(i32, out);
            k_repeat<<<DSC_GPU_BLOCKS(n), DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data,
                                                                     n, repeats,
                                                                     axis_idx, params);
            break;
        }
        case BF16: {
            DSC_DATA(bf16, x);
            DSC_DATA(bf16, out);
            k_repeat<<<DSC_GPU_BLOCKS(n), DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data,
                                                                     n, repeats,
                                                                     axis_idx, params);
            break;
        }
        case F32: {
            DSC_DATA(f32, x);
            DSC_DATA(f32, out);
            k_repeat<<<DSC_GPU_BLOCKS(n), DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data,
                                                                     n, repeats,
                                                                     axis_idx, params);
            break;
        }
        case F64: {
            DSC_DATA(f64, x);
            DSC_DATA(f64, out);
            k_repeat<<<DSC_GPU_BLOCKS(n), DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data,
                                                                     n, repeats,
                                                                     axis_idx, params);
            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

template<typename T>
static DSC_GPU_KERNEL void k_randn(gpu_rand_state *state,
                                   T *DSC_RESTRICT x,
                                   const int n) {
    DSC_GPU_TID();
    DSC_GPU_STRIDE();

    gpu_rand_state s = state[tid];

    for (int i = tid; i < n; i += stride) {
        if constexpr (dsc_is_type<T, f32>() || dsc_is_type<T, bf16>()) {
            x[i] = gpu_sample_normalf(&s);
        } else if constexpr (dsc_is_type<T, f64>()) {
            x[i] = gpu_sample_normal(&s);
        } else {
            static_assert("k_randn - dtype must be real");
        }
    }

    state[tid] = s;
}

void dsc_gpu_randn(dsc_device *dev, dsc_tensor *DSC_RESTRICT x) {
    const dsc_gpu_dev_info *info = (dsc_gpu_dev_info *) dev->extra_info;

    switch (x->dtype) {
        case BF16: {
            DSC_DATA(bf16, x);
            k_randn<bf16><<<1, DSC_GPU_DEFAULT_THREADS>>>(info->rand_state, x_data, x->ne);
            break;
        }
        case F32: {
            DSC_DATA(f32, x);
            k_randn<f32><<<1, DSC_GPU_DEFAULT_THREADS>>>(info->rand_state, x_data, x->ne);
            break;
        }
        case F64: {
            DSC_DATA(f64, x);
            k_randn<f64><<<1, DSC_GPU_DEFAULT_THREADS>>>(info->rand_state, x_data, x->ne);
            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

struct concat_x_params {
    int shape[DSC_MAX_DIMS]{}, stride[DSC_MAX_DIMS]{};
    void *data{};
};

struct concat_out_params {
    int shape[DSC_MAX_DIMS]{};
};

template<typename T>
static DSC_GPU_KERNEL void k_concat(T *DSC_RESTRICT out,
                                    const int n_tensors,
                                    const int n,
                                    const int axis_idx,
                                    const concat_out_params out_params,
                                    const concat_x_params *x_params) {
    DSC_GPU_TID();
    DSC_GPU_STRIDE();

    if (tid >= n) return;

    int out_idx[DSC_MAX_DIMS]{}, x_idx[DSC_MAX_DIMS]{};
    for (int i = tid; i < n; i += stride) {
        // Find the index in the output tensor
        compute_idx_from_linear(out_idx, i, out_params.shape);

        // Based on the index along axis_idx we can determine the input tensor
        const int idx = out_idx[axis_idx];
        int x_tensor_idx = -1;
        int offset = 0;
        for (int x_i = 0; x_i < n_tensors; ++x_i) {
            const int x_dim = x_params[x_i].shape[axis_idx];
            if (idx - offset < x_dim) {
                x_tensor_idx = x_i;
                break;
            }
            offset += x_dim;
        }

        // x_idx then is simply out_idx except along axis_idx where it becomes idx - offset
        for (int dim_i = 0; dim_i < DSC_MAX_DIMS; ++dim_i)
            x_idx[dim_i] = dim_i == axis_idx ? (idx - offset) : out_idx[dim_i];

        T *DSC_RESTRICT x_data = (T *) x_params[x_tensor_idx].data;
        out[i] = x_data[compute_linear_idx(x_idx, x_params[x_tensor_idx].stride)];
    }
}

void dsc_gpu_concat(dsc_device *dev,
                    dsc_tensor **to_concat,
                    const int tensors,
                    dsc_tensor *DSC_RESTRICT out,
                    const int axis_idx) {
    concat_x_params *tmp_params = (concat_x_params *) alloca(tensors * sizeof(concat_x_params));
    for (int i = 0; i < tensors; ++i) {
        tmp_params[i].data = to_concat[i]->buf->data;
        for (int dim_i = 0; dim_i < DSC_MAX_DIMS; ++dim_i) {
            tmp_params[i].shape[dim_i] = to_concat[i]->shape[dim_i];
            tmp_params[i].stride[dim_i] = to_concat[i]->stride[dim_i];
        }
    }
    dsc_data_buffer *x_buf = dsc_data_alloc(dev, tensors * sizeof(concat_x_params));
    concat_x_params *x_params = (concat_x_params *) x_buf->data;
    DSC_GPU_CHECK(gpu_memcpy(x_params, tmp_params, tensors * sizeof(concat_x_params), gpu_memcpy_host_2_device));

    concat_out_params out_params{};
    memcpy(out_params.shape, out->shape, DSC_MAX_DIMS * sizeof(*out_params.shape));

    const int n = out->ne;
    switch (out->dtype) {
        case BOOL: {
            DSC_DATA(bool, out);
            k_concat<<<DSC_GPU_BLOCKS(n),
                       DSC_GPU_DEFAULT_THREADS>>>(out_data, tensors, n, axis_idx, out_params, x_params);
            break;
        }
        case I32: {
            DSC_DATA(i32, out);
            k_concat<<<DSC_GPU_BLOCKS(n),
                       DSC_GPU_DEFAULT_THREADS>>>(out_data, tensors, n, axis_idx, out_params, x_params);
            break;
        }
        case BF16: {
            DSC_DATA(bf16, out);
            k_concat<<<DSC_GPU_BLOCKS(n),
                       DSC_GPU_DEFAULT_THREADS>>>(out_data, tensors, n, axis_idx, out_params, x_params);
            break;
        }
        case F32: {
            DSC_DATA(f32, out);
            k_concat<<<DSC_GPU_BLOCKS(n),
                       DSC_GPU_DEFAULT_THREADS>>>(out_data, tensors, n, axis_idx, out_params, x_params);
            break;
        }
        case F64: {
            DSC_DATA(f64, out);
            k_concat<<<DSC_GPU_BLOCKS(n),
                       DSC_GPU_DEFAULT_THREADS>>>(out_data, tensors, n, axis_idx, out_params, x_params);
            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }
    dsc_data_free(dev, x_buf);
}

struct transpose_params {
    int new_shape[DSC_MAX_DIMS]{}, new_stride[DSC_MAX_DIMS]{};
};

template<typename T>
static DSC_GPU_KERNEL void k_transpose(const T *DSC_RESTRICT x,
                                       T *DSC_RESTRICT out,
                                       const int n,
                                       const transpose_params params) {
    DSC_GPU_TID();
    DSC_GPU_STRIDE();

    if (tid >= n) return;

    int x_idx[DSC_MAX_DIMS]{};
    for (int i = tid; i < n; i += stride) {
        compute_idx_from_linear(x_idx, i, params.new_shape);
        out[i] = x[compute_linear_idx(x_idx, params.new_stride)];
    }
}

void dsc_gpu_transpose(dsc_device *,
                       const dsc_tensor *DSC_RESTRICT x,
                       dsc_tensor *DSC_RESTRICT out,
                       const int *new_shape,
                       const int *new_stride) {
    transpose_params params{};
    memcpy(params.new_shape, new_shape, DSC_MAX_DIMS * sizeof(*params.new_shape));
    memcpy(params.new_stride, new_stride, DSC_MAX_DIMS * sizeof(*params.new_stride));

    const int n = out->ne;
    switch (out->dtype) {
        case BOOL: {
            DSC_DATA(bool, x);
            DSC_DATA(bool, out);
            k_transpose<bool><<<DSC_GPU_BLOCKS(n),
                                DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, n, params);
            break;
        }
        case I32: {
            DSC_DATA(i32, x);
            DSC_DATA(i32, out);
            k_transpose<i32><<<DSC_GPU_BLOCKS(n),
                               DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, n, params);
            break;
        }
        case BF16: {
            DSC_DATA(bf16, x);
            DSC_DATA(bf16, out);
            k_transpose<bf16><<<DSC_GPU_BLOCKS(n),
                               DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, n, params);
            break;
        }
        case F32: {
            DSC_DATA(f32, x);
            DSC_DATA(f32, out);
            k_transpose<f32><<<DSC_GPU_BLOCKS(n),
                               DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, n, params);
            break;
        }
        case F64: {
            DSC_DATA(f64, x);
            DSC_DATA(f64, out);
            k_transpose<f64><<<DSC_GPU_BLOCKS(n),
                               DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, n, params);

            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

struct tril_params {
    int shape[DSC_MAX_DIMS]{};
    int stride[DSC_MAX_DIMS]{};
};

template<typename T>
static DSC_GPU_KERNEL void k_tril(const T *DSC_RESTRICT x,
                                  T *DSC_RESTRICT out,
                                  const int n, const int diagonal,
                                  const tril_params params) {
    DSC_GPU_TID();
    DSC_GPU_STRIDE();

    if (tid >= n) return;

    int idx[DSC_MAX_DIMS];
    for (int i = tid; i < n; i += stride) {
        compute_idx_from_linear(idx, i, params.shape);
        const int row = idx[DSC_MAX_DIMS - 2];
        const int col = idx[DSC_MAX_DIMS - 1];

        out[i] = (col > (row + diagonal)) ? dsc_zero<T>() : x[i];
    }
}

void dsc_gpu_tril(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT x,
                  const int diagonal,
                  dsc_tensor *DSC_RESTRICT out) {
    const int n = x->ne;

    tril_params params;
    // x and out have the same shape and dtype
    memcpy(params.shape, x->shape, DSC_MAX_DIMS * sizeof(*x->shape));
    memcpy(params.stride, x->stride, DSC_MAX_DIMS * sizeof(*x->stride));

    switch (x->dtype) {
        case BOOL: {
            DSC_DATA(bool, x);
            DSC_DATA(bool, out);

            k_tril<<<DSC_GPU_BLOCKS(n),
                     DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, n, diagonal, params);
            break;
        }
        case I32: {
            DSC_DATA(i32, x);
            DSC_DATA(i32, out);

            k_tril<<<DSC_GPU_BLOCKS(n),
                     DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, n, diagonal, params);
            break;
        }
        case BF16: {
            DSC_DATA(bf16, x);
            DSC_DATA(bf16, out);

            k_tril<<<DSC_GPU_BLOCKS(n),
                     DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, n, diagonal, params);
            break;
        }
        case F32: {
            DSC_DATA(f32, x);
            DSC_DATA(f32, out);

            k_tril<<<DSC_GPU_BLOCKS(n),
                     DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, n, diagonal, params);
            break;
        }
        case F64: {
            DSC_DATA(f64, x);
            DSC_DATA(f64, out);

            k_tril<<<DSC_GPU_BLOCKS(n),
                     DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, n, diagonal, params);
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

static DSC_INLINE DSC_GPU_FUNC void compute_slice_from_linear(int *idx_arr, int linear_idx,
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
static DSC_GPU_KERNEL void k_get_slice(const T *DSC_RESTRICT x,
                                       T *DSC_RESTRICT out,
                                       const int n,
                                       const slicing_params params) {
    DSC_GPU_TID();
    DSC_GPU_STRIDE();

    if (tid >= n) return;

    int idx_arr[DSC_MAX_DIMS];

    for (int i = tid; i < n; i += stride) {
        init_slice_idx(idx_arr, params.slices);

        compute_slice_from_linear(idx_arr, i, params.slices, params.shape);
        out[i] = x[compute_linear_idx(idx_arr, params.stride)];
    }
}

static DSC_INLINE void set_slice_params(const dsc_tensor *DSC_RESTRICT x,
                                        const int n_slices,
                                        const dsc_slice *slices,
                                        slicing_params *params) {
    for (int i = 0; i < x->n_dim; ++i) {
        const int dim = dsc_tensor_dim_idx(x, i);
        if (i < n_slices) {
            params->slices[dim] = slices[i];
        } else {
            params->slices[dim] = {0, x->shape[dim], 1};
        }
    }
}

void dsc_gpu_get_slice(dsc_device *,
                       const dsc_tensor *DSC_RESTRICT x,
                       dsc_tensor *DSC_RESTRICT out,
                       const int n_slices, const dsc_slice *slices,
                       const bool whole) {
    if (whole) {
        DSC_DATA(void, x);
        DSC_DATA(void, out);
        DSC_GPU_CHECK(gpu_memcpy(out_data, x_data, out->ne * DSC_DTYPE_SIZE[out->dtype], gpu_memcpy_device_2_device));
        return;
    }

    slicing_params params{};
    memcpy(params.shape, out->shape, DSC_MAX_DIMS * sizeof(*out->shape));
    memcpy(params.stride, x->stride, DSC_MAX_DIMS * sizeof(*x->stride));
    set_slice_params(x, n_slices, slices, &params);

    const int n = out->ne;
    switch (out->dtype) {
        case BOOL: {
            DSC_DATA(bool, x);
            DSC_DATA(bool, out);
            k_get_slice<bool><<<DSC_GPU_BLOCKS(n),
                                DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, n, params);
            break;
        }
        case I32: {
            DSC_DATA(i32, x);
            DSC_DATA(i32, out);
            k_get_slice<i32><<<DSC_GPU_BLOCKS(n),
                               DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, n, params);
            break;
        }
        case BF16: {
            DSC_DATA(bf16, x);
            DSC_DATA(bf16, out);
            k_get_slice<bf16><<<DSC_GPU_BLOCKS(n),
                               DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, n, params);
            break;
        }
        case F32: {
            DSC_DATA(f32, x);
            DSC_DATA(f32, out);
            k_get_slice<f32><<<DSC_GPU_BLOCKS(n),
                               DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, n, params);
            break;
        }
        case F64: {
            DSC_DATA(f64, x);
            DSC_DATA(f64, out);
            k_get_slice<f64><<<DSC_GPU_BLOCKS(n),
                               DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, n, params);
            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }
}

template<typename T>
static DSC_GPU_KERNEL void k_set_slice(T *DSC_RESTRICT xa,
                                       const T *DSC_RESTRICT xb,
                                       const bool xb_scalar,
                                       const int n,
                                       const slicing_params params,
                                       const bool whole) {
    DSC_GPU_TID();
    DSC_GPU_STRIDE();

    if (tid >= n) return;

    if (whole) {
        for (int i = tid; i < n; i += stride) {
            xa[i] = xb_scalar ? xb[0] : xb[i];
        }
    } else {
        int idx_arr[DSC_MAX_DIMS];
        for (int i = tid; i < n; i += stride) {
            init_slice_idx(idx_arr, params.slices);
            compute_slice_from_linear(idx_arr, i, params.slices, params.shape);
            xa[compute_linear_idx(idx_arr, params.stride)] = xb_scalar ? xb[0] : xb[i];
        }
    }
}

void dsc_gpu_set_slice(dsc_device *,
                       dsc_tensor *DSC_RESTRICT xa,
                       const bool xa_scalar,
                       const dsc_tensor *DSC_RESTRICT xb,
                       const bool xb_scalar,
                       const int n_slices, const dsc_slice *slices,
                       const bool whole) {
    if (xa_scalar) {
        int offset = 0;
        for (int i = 0; i < n_slices; ++i)
            offset += (slices[i].start * dsc_tensor_get_stride(xa, i));

        DSC_DATA(byte, xa);
        DSC_DATA(void, xb);
        DSC_GPU_CHECK(gpu_memcpy(xa_data + (offset * DSC_DTYPE_SIZE[xa->dtype]),
                                 xb_data, DSC_DTYPE_SIZE[xa->dtype], gpu_memcpy_device_2_device));
    } else {
        slicing_params params{};
        memcpy(params.stride, xa->stride, DSC_MAX_DIMS * sizeof(*xa->stride));
        set_slice_params(xa, n_slices, slices, &params);

        int n = 1;
        for (int i = 0; i < xa->n_dim; ++i) {
            const int dim = dsc_tensor_dim_idx(xa, i);
            const int ne_i = abs(params.slices[dim].start - params.slices[dim].stop);
            const int step_i = abs(params.slices[dim].step);
            params.shape[dim] = (ne_i + step_i - 1) / step_i;
            n *= params.shape[dim];
        }

        switch (xa->dtype) {
            case BOOL: {
                DSC_DATA(bool, xa);
                DSC_DATA(bool, xb);
                k_set_slice<bool><<<DSC_GPU_BLOCKS(n),
                                    DSC_GPU_DEFAULT_THREADS>>>(xa_data, xb_data, xb_scalar, n, params, whole);
                break;
            }
            case I32: {
                DSC_DATA(i32, xa);
                DSC_DATA(i32, xb);
                k_set_slice<i32><<<DSC_GPU_BLOCKS(n),
                                   DSC_GPU_DEFAULT_THREADS>>>(xa_data, xb_data, xb_scalar, n, params, whole);
                break;
            }
            case BF16: {
                DSC_DATA(bf16, xa);
                DSC_DATA(bf16, xb);
                k_set_slice<bf16><<<DSC_GPU_BLOCKS(n),
                                   DSC_GPU_DEFAULT_THREADS>>>(xa_data, xb_data, xb_scalar, n, params, whole);
                break;
            }
            case F32: {
                DSC_DATA(f32, xa);
                DSC_DATA(f32, xb);
                k_set_slice<f32><<<DSC_GPU_BLOCKS(n),
                                   DSC_GPU_DEFAULT_THREADS>>>(xa_data, xb_data, xb_scalar, n, params, whole);
                break;
            }
            case F64: {
                DSC_DATA(f64, xa);
                DSC_DATA(f64, xb);
                k_set_slice<f64><<<DSC_GPU_BLOCKS(n),
                                   DSC_GPU_DEFAULT_THREADS>>>(xa_data, xb_data, xb_scalar, n, params, whole);
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
    int n_dim{};
};

template<typename Tout, typename Tx = Tout, typename Op,
         bool xa_scalar, bool xb_scalar, bool shape_matches>
static DSC_GPU_KERNEL void k_binary_op(const Tx *xa, const Tx *xb, Tout *out,
                                       const int n, Op op,
                                       const binary_params params = {}) {
    DSC_GPU_TID();
    DSC_GPU_STRIDE();

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

template<typename Tout, typename Tx = Tout, typename Op>
static DSC_INLINE void binary_op(const dsc_tensor *xa,
                                 const dsc_tensor *xb,
                                 dsc_tensor *out,
                                 Op op) {
    const bool same_shape = xa->n_dim == xb->n_dim && memcmp(&dsc_tensor_get_dim(xa, 0),
                                                             &dsc_tensor_get_dim(xb, 0),
                                                             xa->n_dim * sizeof(*xa->shape)) == 0;
    DSC_DATA_ALIAS(Tx, xa);
    DSC_DATA_ALIAS(Tx, xb);
    DSC_DATA_ALIAS(Tout, out);

    const int n = out->ne;

    if (dsc_is_scalar(xa)) {
        k_binary_op<Tout, Tx, Op, true, false, false><<<DSC_GPU_BLOCKS(n),
                                                        DSC_GPU_DEFAULT_THREADS>>>(xa_data,
                                                                                   xb_data,
                                                                                   out_data,
                                                                                   n,
                                                                                   op);
    } else if (dsc_is_scalar(xb)) {
        k_binary_op<Tout, Tx, Op, false, true, false><<<DSC_GPU_BLOCKS(n),
                                                        DSC_GPU_DEFAULT_THREADS>>>(xa_data,
                                                                                   xb_data,
                                                                                   out_data,
                                                                                   n,
                                                                                   op);
    } else if (same_shape) {
        k_binary_op<Tout, Tx, Op, false, false, true><<<DSC_GPU_BLOCKS(n),
                                                        DSC_GPU_DEFAULT_THREADS>>>(xa_data,
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

        k_binary_op<Tout, Tx, Op, false, false, false><<<DSC_GPU_BLOCKS(n),
                                                         DSC_GPU_DEFAULT_THREADS>>>(xa_data,
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
        case BOOL:
            if constexpr (is_comparison_op<Op>()) {
                switch (xa->dtype) {
                    case BOOL:
                        binary_op<bool, bool>(xa, xb, out, op);
                        break;
                    case I32:
                        binary_op<bool, i32>(xa, xb, out, op);
                        break;
                    case BF16:
                        binary_op<bool, bf16>(xa, xb, out, op);
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
        case BF16:
            binary_op<bf16>(xa, xb, out, op);
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

void dsc_gpu_add(dsc_device *,
                 const dsc_tensor *xa,
                 const dsc_tensor *xb,
                 dsc_tensor *out) {
    binary_op(xa, xb, out, gpu_add_op());
}

void dsc_gpu_sub(dsc_device *,
                 const dsc_tensor *xa,
                 const dsc_tensor *xb,
                 dsc_tensor *out) {
    binary_op(xa, xb, out, gpu_sub_op());
}

void dsc_gpu_mul(dsc_device *,
                 const dsc_tensor *xa,
                 const dsc_tensor *xb,
                 dsc_tensor *out) {
    binary_op(xa, xb, out, gpu_mul_op());
}

void dsc_gpu_div(dsc_device *,
                 const dsc_tensor *xa,
                 const dsc_tensor *xb,
                 dsc_tensor *out) {
    binary_op(xa, xb, out, gpu_div_op());
}

void dsc_gpu_pow(dsc_device *,
                 const dsc_tensor *xa,
                 const dsc_tensor *xb,
                 dsc_tensor *out) {
    binary_op(xa, xb, out, gpu_pow_op());
}

template<typename T>
static DSC_INLINE void gemm_op(const gpu_blas_handle handle,
                               const dsc_tensor *DSC_RESTRICT xa,
                               const dsc_tensor *DSC_RESTRICT xb,
                               const bool trans_b,
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

    DSC_DATA(T, xa);
    DSC_DATA(T, xb);
    DSC_DATA(T, out);

    const T alpha = 1, beta = 0;
    const gpu_blas_op a_op = trans_b ? GPU_BLAS_OP_T : GPU_BLAS_OP_N;

    for (int d0 = 0; d0 < d0_out; ++d0) {
        for (int d1 = 0; d1 < d1_out; ++d1) {
            const int out_offset = d0 * out->stride[0] + d1 * out->stride[1];
            const int xa_offset = d0 * xa_stride_d0 + d1 * xa_stride_d1;
            const int xb_offset = d0 * xb_stride_d0 + d1 * xb_stride_d1;

            if constexpr (dsc_is_type<T, f64>()) {
                DSC_GPU_BLAS_CHECK(gpu_blas_dgemm(handle,
                                                  a_op, GPU_BLAS_OP_N, n, m, k,
                                                  &alpha, &xb_data[xb_offset], stride_b,
                                                  &xa_data[xa_offset], stride_a, &beta,
                                                  &out_data[out_offset], stride_out));
            } else if constexpr (dsc_is_type<T, f32>()) {
                DSC_GPU_BLAS_CHECK(gpu_blas_sgemm(handle,
                                                  a_op, GPU_BLAS_OP_N, n, m, k,
                                                  &alpha, &xb_data[xb_offset], stride_b,
                                                  &xa_data[xa_offset], stride_a, &beta,
                                                  &out_data[out_offset], stride_out));
            } else if constexpr (dsc_is_type<T, bf16>()) {
                DSC_GPU_BLAS_CHECK(gpu_blas_bfgemm(handle,
                                                   a_op, GPU_BLAS_OP_N, n, m, k,
                                                   &alpha, &xb_data[xb_offset], GPU_GEMM_DTYPE_BF16, stride_b,
                                                   &xa_data[xa_offset], GPU_GEMM_DTYPE_BF16, stride_a, &beta,
                                                   &out_data[out_offset], GPU_GEMM_DTYPE_BF16, stride_out,
                                                   &out_data[out_offset], GPU_GEMM_DTYPE_BF16, stride_out,
                                                   GPU_GEMM_DTYPE_BF16, GPU_GEMM_ALGO, 0, 0));
            } else {
                static_assert("T must be real");
            }
        }
    }
}

void dsc_gpu_matmul(dsc_device *dev,
                    const dsc_tensor *DSC_RESTRICT xa,
                    const dsc_tensor *DSC_RESTRICT xb,
                    const bool trans_b,
                    dsc_tensor *DSC_RESTRICT out) {
    const dsc_gpu_dev_info *info = (dsc_gpu_dev_info *) dev->extra_info;

    switch (xa->dtype) {
        case BF16: {
            gemm_op<bf16>(info->blas_handle, xa, xb, trans_b, out);
            break;
        }
        case F32: {
            gemm_op<f32>(info->blas_handle, xa, xb, trans_b, out);
            break;
        }
        case F64: {
            gemm_op<f64>(info->blas_handle, xa, xb, trans_b, out);
            break;
        }
        DSC_INVALID_CASE("unsupported dtype=%d", xa->dtype);
    }
}

void dsc_gpu_compare(dsc_device *,
                     const dsc_tensor *xa,
                     const dsc_tensor *xb,
                     const dsc_comparison_op comp,
                     dsc_tensor *out) {
    switch (comp) {
        case EQ:
            binary_op(xa, xb, out, gpu_eq_op());
            break;
        case NE:
            binary_op(xa, xb, out, gpu_ne_op());
            break;
        case LT:
            binary_op(xa, xb, out, gpu_lt_op());
            break;
        case LE:
            binary_op(xa, xb, out, gpu_le_op());
            break;
        case GT:
            binary_op(xa, xb, out, gpu_gt_op());
            break;
        case GE:
            binary_op(xa, xb, out, gpu_ge_op());
            break;
        DSC_INVALID_CASE("unknown comparison=%d", comp);
    }
}

struct masked_fill_params {
    int mask_shape[DSC_MAX_DIMS]{};
    int mask_stride[DSC_MAX_DIMS]{};
};

template<typename T>
static DSC_GPU_KERNEL void k_masked_fill(T *DSC_RESTRICT x,
                                         const bool *DSC_RESTRICT mask,
                                         const int n,
                                         const T value,
                                         const masked_fill_params params) {
    DSC_GPU_TID();
    DSC_GPU_STRIDE();

    if (tid >= n) return;

    int mask_idx[DSC_MAX_DIMS];

    for (int i = tid; i < n; i += stride) {
        compute_idx_from_linear(mask_idx, i, params.mask_shape);
        x[i] = mask[compute_linear_idx(mask_idx, params.mask_stride)] ? value : x[i];
    }
}

void dsc_gpu_masked_fill(dsc_device *,
                         dsc_tensor *DSC_RESTRICT x,
                         const dsc_tensor *DSC_RESTRICT mask,
                         const f64 value) {
    const int n = x->ne;

    masked_fill_params params;
    memcpy(params.mask_shape, mask->shape, DSC_MAX_DIMS * sizeof(*mask->shape));
    memcpy(params.mask_stride, mask->stride, DSC_MAX_DIMS * sizeof(*mask->stride));


    switch (x->dtype) {
        case BOOL: {
            DSC_DATA(bool, x);
            DSC_DATA(bool, mask);

            k_masked_fill<<<DSC_GPU_BLOCKS(n),
                            DSC_GPU_DEFAULT_THREADS>>>(x_data, mask_data, n,
                                                       (bool) value, params);
            break;
        }
        case I32: {
            DSC_DATA(i32, x);
            DSC_DATA(bool, mask);

            k_masked_fill<<<DSC_GPU_BLOCKS(n),
                            DSC_GPU_DEFAULT_THREADS>>>(x_data, mask_data, n,
                                                       (i32) value, params);
            break;
        }
        case BF16: {
            DSC_DATA(bf16, x);
            DSC_DATA(bool, mask);

            k_masked_fill<<<DSC_GPU_BLOCKS(n),
                            DSC_GPU_DEFAULT_THREADS>>>(x_data, mask_data, n,
                                                       (bf16) value, params);
            break;
        }
        case F32: {
            DSC_DATA(f32, x);
            DSC_DATA(bool, mask);

            k_masked_fill<<<DSC_GPU_BLOCKS(n),
                            DSC_GPU_DEFAULT_THREADS>>>(x_data, mask_data, n,
                                                       (f32) value, params);
            break;
        }
        case F64: {
            DSC_DATA(f64, x);
            DSC_DATA(bool, mask);

            k_masked_fill<<<DSC_GPU_BLOCKS(n),
                            DSC_GPU_DEFAULT_THREADS>>>(x_data, mask_data, n,
                                                       value, params);
            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

struct outer_params {
    int shape[DSC_MAX_DIMS]{};
};

template<typename T>
static DSC_GPU_KERNEL void k_outer(const T *DSC_RESTRICT xa,
                                   const T *DSC_RESTRICT xb,
                                   T *DSC_RESTRICT out,
                                   const int n,
                                   const outer_params params) {
    DSC_GPU_TID();
    DSC_GPU_STRIDE();

    if (tid >= n) return;

    int idx[DSC_MAX_DIMS]{};
    for (int i = tid; i < n; i += stride) {
        // Compute the index in the output tensor
        compute_idx_from_linear(idx, i, params.shape);
        // Since xa and xb are assumed to be 1D I can get the ij indices from the computed idx
        const int ii = idx[DSC_MAX_DIMS - 2];
        const int jj = idx[DSC_MAX_DIMS - 1];

        out[i] = gpu_mul_op()(xa[ii], xb[jj]);
    }
}

void dsc_gpu_outer(dsc_device *,
                   const dsc_tensor *DSC_RESTRICT xa,
                   const dsc_tensor *DSC_RESTRICT xb,
                   dsc_tensor *DSC_RESTRICT out) {
    const int n = out->ne;

    outer_params params;
    memcpy(params.shape, out->shape, DSC_MAX_DIMS * sizeof(*out->shape));

    switch (xa->dtype) {
        case BOOL: {
            DSC_DATA(bool, xa);
            DSC_DATA(bool, xb);
            DSC_DATA(bool, out);

            k_outer<<<DSC_GPU_BLOCKS(n),
                      DSC_GPU_DEFAULT_THREADS>>>(xa_data, xb_data, out_data, n, params);
            break;
        }
        case I32: {
            DSC_DATA(i32, xa);
            DSC_DATA(i32, xb);
            DSC_DATA(i32, out);

            k_outer<<<DSC_GPU_BLOCKS(n),
                      DSC_GPU_DEFAULT_THREADS>>>(xa_data, xb_data, out_data, n, params);
            break;
        }
        case BF16: {
            DSC_DATA(bf16, xa);
            DSC_DATA(bf16, xb);
            DSC_DATA(bf16, out);

            k_outer<<<DSC_GPU_BLOCKS(n),
                      DSC_GPU_DEFAULT_THREADS>>>(xa_data, xb_data, out_data, n, params);
            break;
        }
        case F32: {
            DSC_DATA(f32, xa);
            DSC_DATA(f32, xb);
            DSC_DATA(f32, out);

            k_outer<<<DSC_GPU_BLOCKS(n),
                      DSC_GPU_DEFAULT_THREADS>>>(xa_data, xb_data, out_data, n, params);
            break;
        }
        case F64: {
            DSC_DATA(f64, xa);
            DSC_DATA(f64, xb);
            DSC_DATA(f64, out);

            k_outer<<<DSC_GPU_BLOCKS(n),
                      DSC_GPU_DEFAULT_THREADS>>>(xa_data, xb_data, out_data, n, params);
            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", xa->dtype);
    }
}

struct where_params {
    int cond_shape[DSC_MAX_DIMS], cond_stride[DSC_MAX_DIMS];
    int input_shape[DSC_MAX_DIMS], input_stride[DSC_MAX_DIMS];
    int other_shape[DSC_MAX_DIMS], other_stride[DSC_MAX_DIMS];
};

template<typename T>
static DSC_GPU_KERNEL void k_where(const bool *DSC_RESTRICT condition,
                                   const T *DSC_RESTRICT input,
                                   const T *DSC_RESTRICT other,
                                   T *DSC_RESTRICT out,
                                   const int n,
                                   const where_params params) {
    DSC_GPU_TID();
    DSC_GPU_STRIDE();

    if (tid >= n) return;

    int cond_idx_arr[DSC_MAX_DIMS], input_idx_arr[DSC_MAX_DIMS], other_idx_arr[DSC_MAX_DIMS];
    for (int i = tid; i < n; i += stride) {
        compute_idx_from_linear(cond_idx_arr, i, params.cond_shape);
        compute_idx_from_linear(input_idx_arr, i, params.input_shape);
        compute_idx_from_linear(other_idx_arr, i, params.other_shape);
        const int cond_idx = compute_linear_idx(cond_idx_arr, params.cond_stride);
        const int input_idx = compute_linear_idx(input_idx_arr, params.input_stride);
        const int other_idx = compute_linear_idx(other_idx_arr, params.other_stride);

        out[i] = condition[cond_idx] ? input[input_idx] : other[other_idx];
    }
}

void dsc_gpu_where(dsc_device *,
                   const dsc_tensor *DSC_RESTRICT condition,
                   const dsc_tensor *DSC_RESTRICT input,
                   const dsc_tensor *DSC_RESTRICT other,
                   dsc_tensor *DSC_RESTRICT out) {
    DSC_DATA(bool, condition);

    // Prepare params
    where_params params{};
    memcpy(params.cond_shape, condition->shape, DSC_MAX_DIMS * sizeof(*condition->shape));
    memcpy(params.cond_stride, condition->stride, DSC_MAX_DIMS * sizeof(*condition->stride));
    memcpy(params.input_shape, input->shape, DSC_MAX_DIMS * sizeof(*input->shape));
    memcpy(params.input_stride, input->stride, DSC_MAX_DIMS * sizeof(*input->stride));
    memcpy(params.other_shape, other->shape, DSC_MAX_DIMS * sizeof(*other->shape));
    memcpy(params.other_stride, other->stride, DSC_MAX_DIMS * sizeof(*other->stride));

    const int n = out->ne;
    switch (out->dtype) {
        case BOOL: {
            DSC_DATA(bool, input);
            DSC_DATA(bool, other);
            DSC_DATA(bool, out);

            k_where<<<DSC_GPU_BLOCKS(n),
                      DSC_GPU_DEFAULT_THREADS>>>(condition_data, input_data,
                                                 other_data, out_data, n, params);
            break;
        }
        case I32: {
            DSC_DATA(i32, input);
            DSC_DATA(i32, other);
            DSC_DATA(i32, out);

            k_where<<<DSC_GPU_BLOCKS(n),
                      DSC_GPU_DEFAULT_THREADS>>>(condition_data, input_data,
                                                 other_data, out_data, n, params);
            break;
        }
        case BF16: {
            DSC_DATA(bf16, input);
            DSC_DATA(bf16, other);
            DSC_DATA(bf16, out);

            k_where<<<DSC_GPU_BLOCKS(n),
                      DSC_GPU_DEFAULT_THREADS>>>(condition_data, input_data,
                                                 other_data, out_data, n, params);
            break;
        }
        case F32: {
            DSC_DATA(f32, input);
            DSC_DATA(f32, other);
            DSC_DATA(f32, out);

            k_where<<<DSC_GPU_BLOCKS(n),
                      DSC_GPU_DEFAULT_THREADS>>>(condition_data, input_data,
                                                 other_data, out_data, n, params);
            break;
        }
        case F64: {
            DSC_DATA(f64, input);
            DSC_DATA(f64, other);
            DSC_DATA(f64, out);

            k_where<<<DSC_GPU_BLOCKS(n),
                      DSC_GPU_DEFAULT_THREADS>>>(condition_data, input_data,
                                                 other_data, out_data, n, params);
            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }
}

// ============================================================
// Unary Operations

template<typename Tx, typename To = Tx, typename Op>
static DSC_GPU_KERNEL void k_unary_op(const Tx *DSC_RESTRICT x,
                                      To *DSC_RESTRICT out,
                                      const int n, Op op) {
    DSC_GPU_TID();
    DSC_GPU_STRIDE();

    for (int i = tid; i < n; i += stride) {
        out[i] = op(x[i]);
    }
}

template<typename Op>
static DSC_INLINE void unary_op(const dsc_tensor *DSC_RESTRICT x,
                                dsc_tensor *DSC_RESTRICT out, Op op) {
    const int n = x->ne;
    switch (x->dtype) {
        case BF16: {
            DSC_DATA(bf16, x);
            DSC_DATA(bf16, out);
            k_unary_op<bf16><<<DSC_GPU_BLOCKS(n),
                              DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, n, op);
            break;
        }
        case F32: {
            DSC_DATA(f32, x);
            DSC_DATA(f32, out);
            k_unary_op<f32><<<DSC_GPU_BLOCKS(n),
                              DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, n, op);
            break;
        }
        case F64: {
            DSC_DATA(f64, x);
            DSC_DATA(f64, out);
            k_unary_op<f64><<<DSC_GPU_BLOCKS(n),
                              DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, n, op);
            break;
        }
        DSC_INVALID_CASE("unknown dtype=%d", x->dtype);
    }
}

void dsc_gpu_cos(dsc_device *,
                 const dsc_tensor *DSC_RESTRICT x,
                 dsc_tensor *DSC_RESTRICT out) {
    unary_op(x, out, gpu_cos_op());
}

void dsc_gpu_sin(dsc_device *,
                 const dsc_tensor *DSC_RESTRICT x,
                 dsc_tensor *DSC_RESTRICT out) {
    unary_op(x, out, gpu_sin_op());
}

void dsc_gpu_tanh(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out) {
    unary_op(x, out, gpu_tanh_op());
}

void dsc_gpu_exp(dsc_device *,
                 const dsc_tensor *DSC_RESTRICT x,
                 dsc_tensor *DSC_RESTRICT out) {
    unary_op(x, out, gpu_exp_op());
}

void dsc_gpu_sqrt(dsc_device *,
                  const dsc_tensor *DSC_RESTRICT x,
                  dsc_tensor *DSC_RESTRICT out) {
    unary_op(x, out, gpu_sqrt_op());
}

// ============================================================
// Unary Operations Along Axis

struct reduce_params {
    int x_stride[DSC_MAX_DIMS]{}, out_shape[DSC_MAX_DIMS]{};
    int axis_idx{}, axis_n{}, shift{};
};

template<typename T, typename Op, typename AOp>
static DSC_GPU_KERNEL void k_reduce(const T *DSC_RESTRICT x,
                                    T *DSC_RESTRICT out,
                                    const T initial_value,
                                    const int n,
                                    Op reduction_op,
                                    AOp atomic_reduction_op,
                                    const reduce_params params) {
    int out_idx[DSC_MAX_DIMS]{}, x_idx[DSC_MAX_DIMS]{};

    for (int reduction_idx = (int) blockIdx.x; reduction_idx < n; reduction_idx += (int) gridDim.x) {
        if (threadIdx.x == 0) out[reduction_idx] = initial_value;
        __syncthreads();

        compute_idx_from_linear(out_idx, reduction_idx, params.out_shape);
        T partial_result = initial_value;
        for (int i = (int) threadIdx.x; i < params.axis_n; i += (int) blockDim.x) {
            // Get the index in the input array
            for (int dim_idx = 0; dim_idx < DSC_MAX_DIMS; ++dim_idx) {
                if (dim_idx > params.axis_idx) x_idx[dim_idx] = out_idx[dim_idx];
                else if (dim_idx == params.axis_idx)
                    x_idx[dim_idx] = i;
                else
                    x_idx[dim_idx] = out_idx[dim_idx + params.shift];
            }
            partial_result = reduction_op(partial_result, x[compute_linear_idx(x_idx, params.x_stride)]);
        }
        atomic_reduction_op(&out[reduction_idx], partial_result);
    }
}

static DSC_INLINE void init_reduction_params(const dsc_tensor *DSC_RESTRICT x,
                                             const dsc_tensor *DSC_RESTRICT out,
                                             const int axis_idx,
                                             reduce_params *params) {
    params->axis_idx = axis_idx;
    params->axis_n = x->shape[axis_idx];
    params->shift = out->n_dim < x->n_dim;
    memcpy(params->x_stride, x->stride, DSC_MAX_DIMS * sizeof(*x->stride));
    memcpy(params->out_shape, out->shape, DSC_MAX_DIMS * sizeof(*out->shape));
}

template<typename T, typename Op, typename AOp>
static DSC_INLINE void reduce_op(const dsc_tensor *DSC_RESTRICT x,
                                 dsc_tensor *DSC_RESTRICT out,
                                 const int axis_idx,
                                 const T default_value,
                                 Op op,
                                 AOp atomic_op) {
    DSC_DATA(T, x);
    DSC_DATA(T, out);

    reduce_params params;
    init_reduction_params(x, out, axis_idx, &params);

    const int n = out->ne;

    k_reduce<<<DSC_GPU_BLOCKS(n),
               DSC_GPU_DEFAULT_THREADS>>>(x_data, out_data, default_value,
                                          n, op, atomic_op, params);
}

// TODO: not clear how to do reductions with bf16 (atomics?)
void dsc_gpu_sum(dsc_device *,
                 const dsc_tensor *DSC_RESTRICT x,
                 dsc_tensor *DSC_RESTRICT out,
                 const int axis_idx) {
    switch (out->dtype) {
        case F32:
            reduce_op<f32, gpu_add_op, gpu_atomic_add_op>(x, out, axis_idx, dsc_zero<f32>(),
                                                          gpu_add_op(), gpu_atomic_add_op());
            break;
        case F64:
            reduce_op<f64, gpu_add_op, gpu_atomic_add_op>(x, out, axis_idx, dsc_zero<f64>(),
                                                          gpu_add_op(), gpu_atomic_add_op());
            break;
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }
}

void dsc_gpu_min(dsc_device *,
                 const dsc_tensor *DSC_RESTRICT x,
                 dsc_tensor *DSC_RESTRICT out,
                 const int axis_idx) {
    switch (out->dtype) {
        case F32:
            reduce_op<f32, gpu_min_op, gpu_atomic_min_op>(x, out, axis_idx, dsc_inf<f32, true>(),
                                                          gpu_min_op(), gpu_atomic_min_op());
            break;
        case F64:
            reduce_op<f64, gpu_min_op, gpu_atomic_min_op>(x, out, axis_idx, dsc_inf<f64, true>(),
                                                          gpu_min_op(), gpu_atomic_min_op());
            break;
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }
}

void dsc_gpu_max(dsc_device *,
                 const dsc_tensor *DSC_RESTRICT x,
                 dsc_tensor *DSC_RESTRICT out,
                 const int axis_idx) {
    switch (out->dtype) {
        case F32:
            reduce_op<f32, gpu_max_op, gpu_atomic_max_op>(x, out, axis_idx, dsc_inf<f32, false>(),
                                                          gpu_max_op(), gpu_atomic_max_op());
            break;
        case F64:
            reduce_op<f64, gpu_max_op, gpu_atomic_max_op>(x, out, axis_idx, dsc_inf<f64, false>(),
                                                          gpu_max_op(), gpu_atomic_max_op());
            break;
        DSC_INVALID_CASE("unknown dtype=%d", out->dtype);
    }
}