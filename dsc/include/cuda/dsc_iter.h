// Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "dsc.h"
#include "cuda/dsc_cuda.h"

// TODO: (2)

struct dsc_slice_iterator {
    DSC_CUDA_FUNC dsc_slice_iterator(const int *shape, const int *stride, const int n_dim,
                                     const int n_slices, const dsc_slice *slices) :
            shape_(shape), stride_(stride), n_dim_(n_dim) {
        for (int i = 0; i < n_dim; ++i) {
            const int dim_idx = (DSC_MAX_DIMS - n_dim_) + i;
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

    DSC_INLINE DSC_CUDA_FUNC bool has_next() const {
        return !end_;
    }

    DSC_INLINE DSC_CUDA_FUNC void advance(const int steps = 1) {
        for (int step = 0; step < steps && !end_; ++step) {
            end_ = true;
            for (int i = DSC_MAX_DIMS - 1; i >= (DSC_MAX_DIMS - n_dim_); --i) {
                idx_[i] += step_[i];
                if ((step_[i] > 0 && idx_[i] < stop_[i]) ||
                    (step_[i] < 0 && idx_[i] > stop_[i])) {
                    end_ = false;
                    break;
                }
                idx_[i] = start_[i];
            }
        }
    }

    DSC_INLINE DSC_CUDA_FUNC int index() const {
        int result = 0;
        for (int i = 0; i < DSC_MAX_DIMS; ++i) {
            result += idx_[i] * stride_[i];
        }
        return result;
    }

private:
    const int *shape_;
    const int *stride_;
    int idx_[DSC_MAX_DIMS]{};
    int start_[DSC_MAX_DIMS]{};
    int stop_[DSC_MAX_DIMS]{};
    int step_[DSC_MAX_DIMS]{};
    const int n_dim_;
    bool end_ = false;
};