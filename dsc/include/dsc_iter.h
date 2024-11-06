// Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "dsc.h"

struct dsc_axis_iterator {
    dsc_axis_iterator(const dsc_tensor *x, const int axis,
                      const int axis_n = -1) noexcept :
            shape_(x->shape), stride_(x->stride),
            axis_(axis), axis_n_((axis_n < 0 || axis_n > x->shape[axis]) ? x->shape[axis] : axis_n) {

    }

    DSC_INLINE void next() noexcept {
        if (++idx_[axis_] < axis_n_) [[likely]] {
            index_ += stride_[axis_];
            return;
        }

        index_ -= (idx_[axis_] - 1) * stride_[axis_];
        idx_[axis_] = 0;

        bool still_left = false;
        for (int i = DSC_MAX_DIMS - 1; i >= 0; --i) {
            if (i == axis_)
                continue;

            if (++idx_[i] < shape_[i]) {
                index_ += stride_[i];
                still_left = true;
                break;
            }

            // Rollover this dimension
            index_ -= (idx_[i] - 1) * stride_[i];
            idx_[i] = 0;
            // If this is the first dimension, and it rolls over we are done
            end_ = i == 0;
        }
        if (axis_ == 0 && !still_left)
            end_ = true;
    }

    DSC_INLINE int index() const noexcept {
        return index_;
    }

    DSC_INLINE bool has_next() const noexcept {
        return !end_;
    }

private:
    int index_ = 0;
    int idx_[DSC_MAX_DIMS]{};
    const int *shape_;
    const int *stride_;
    int axis_;
    int axis_n_;
    bool end_ = false;
};

struct dsc_broadcast_iterator {
    dsc_broadcast_iterator(const dsc_tensor *x, const int *out_shape) noexcept :
            x_shape_(x->shape), x_stride_(x->stride), out_shape_(out_shape) {
        for (int i = 0; i < DSC_MAX_DIMS; ++i) {
            x_broadcast_stride_[i] = x_shape_[i] < out_shape_[i] ? 0 : x_stride_[i];
        }
    }

    DSC_INLINE void next() noexcept {
        for (int i = DSC_MAX_DIMS - 1; i >= 0; --i) {
            if (++x_idx_[i] < out_shape_[i]) [[likely]] {
                index_ += x_broadcast_stride_[i];
                return;
            }
            // Rollover this dimension
            index_ -= (x_idx_[i] - 1) * x_broadcast_stride_[i];
            x_idx_[i] = 0;
        }
    }

    DSC_INLINE int index() const noexcept {
        return index_;
    }

private:
    int index_ = 0;
    const int *x_shape_, *x_stride_, *out_shape_;
    int x_broadcast_stride_[DSC_MAX_DIMS]{}, x_idx_[DSC_MAX_DIMS]{};
};

struct dsc_slice_iterator {
    dsc_slice_iterator(const dsc_tensor *x, const int n_slices, const dsc_slice *slices) noexcept :
            shape_(x->shape), stride_(x->stride), n_dim_(x->n_dim) {
        for (int i = 0; i < x->n_dim; ++i) {
            const int dim_idx = dsc_tensor_dim(x, i);
            if (i < n_slices) {
                start_[dim_idx] = slices[i].start;
                stop_[dim_idx] = slices[i].stop;
                step_[dim_idx] = slices[i].step;
            } else {
                start_[dim_idx] = 0;
                stop_[dim_idx] = shape_[dim_idx];
                step_[dim_idx] = 1;
            }

            idx_[dim_idx] = start_[dim_idx];
        }
    }

    DSC_INLINE bool has_next() const noexcept {
        return !end_;
    }

    DSC_INLINE void next() noexcept {
        for (int i = DSC_MAX_DIMS - 1; i >= (DSC_MAX_DIMS - n_dim_); --i) {
            idx_[i] += step_[i];
            if ((step_[i] > 0 && idx_[i] < stop_[i]) ||
                (step_[i] < 0 && idx_[i] > stop_[i])) [[likely]] {
                return;
            }
            idx_[i] = start_[i];
        }
        end_ = true;
    }

    DSC_INLINE int index() const noexcept {
        return compute_index<DSC_MAX_DIMS, 0>();
    }

private:
    template<int N, int Cur = 0>
    constexpr int compute_index() const noexcept {
        // Note: computing the index on the fly is way easier than keeping track of the current index
        // and increasing/decreasing it after each step, but it requires some benchmarking!
        if constexpr (Cur == N) {
            return 0;
        } else {
            return idx_[Cur] * stride_[Cur] + compute_index<N, Cur + 1>();
        }
    }

    const int *shape_;
    const int *stride_;
    int idx_[DSC_MAX_DIMS]{};
    int start_[DSC_MAX_DIMS]{};
    int stop_[DSC_MAX_DIMS]{};
    int step_[DSC_MAX_DIMS]{};
    const int n_dim_;
    bool end_ = false;
};