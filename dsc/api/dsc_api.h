// Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "dsc.h"
#include <cstring>
#include <initializer_list>
#include <algorithm> // std::copy


#define DSC_SLICE_ALL()                 (dsc_slice{.start = DSC_VALUE_NONE, .stop = DSC_VALUE_NONE, .step = 1})
// Convention to identify a single element instead of a slice
#define DSC_SLICE_IDX(idx_)             (dsc_slice{.start = (idx_), .stop = (idx_), .step = (idx_)})
#define DSC_SLICE_ALL_STEP(step_)       (dsc_slice{.start = DSC_VALUE_NONE, .stop = DSC_VALUE_NONE, .step = (step_)})
#define DSC_SLICE_FROM(start_)          (dsc_slice{.start = (start_), .stop = DSC_VALUE_NONE, .step = 1})
#define DSC_SLICE_TO(stop_)             (dsc_slice{.start = DSC_VALUE_NONE, .stop = (stop_), .step = 1})
#define DSC_SLICE_RANGE(start_, stop_)  (dsc_slice{.start = (start_), .stop = (stop_), .step = 1})


namespace dsc {

static dsc_ctx *ctx = nullptr;

static DSC_INLINE void init(u64 main_mem, u64 scratch_mem = 0) noexcept {
    if (scratch_mem == 0) {
        main_mem = (u64) ((f64) main_mem * 0.9);
        scratch_mem = (u64) ((f64) main_mem * 0.1);
    }
    ctx = dsc_ctx_init(main_mem, scratch_mem);
}

template<typename T>
class tensor {

public:
    // ============================================================
    // Constructors / Destructors

    tensor() noexcept : x_(nullptr) {}

    tensor(dsc_tensor *x) noexcept : x_(x) {}

    tensor(std::initializer_list<T> scalars) noexcept {
        const int shape = scalars.size();
        x_ = dsc_new_tensor(ctx, 1, &shape, dsc_type_mapping<T>::value);
        std::copy(scalars.begin(), scalars.end(), (T *) x_->data);
    }

    tensor(std::initializer_list<int> shape, const T fill) noexcept {
        int shape_arr[DSC_MAX_DIMS];
        std::copy(shape.begin(), shape.end(), shape_arr);
        x_ = dsc_new_tensor(ctx, shape.size(), shape_arr, dsc_type_mapping<T>::value);
        std::fill_n((T *) x_->data, x_->ne, fill);
    }

    tensor(const T *data, const int ne) noexcept {
        x_ = dsc_new_tensor(ctx, 1, &ne, dsc_type_mapping<T>::value);
        memcpy(x_->data, data, ne * sizeof(T));
    }

    tensor(const tensor &other) noexcept {
        x_ = dsc_new_tensor(ctx, other.x_->n_dim, other.x_->shape, other.x_->dtype);
        memcpy(x_->data, other.x_->data, other.x_->ne * DSC_DTYPE_SIZE[other.x_->dtype]);
    }

    tensor(tensor &&other) noexcept : x_(other.x_) {
        other.x_ = nullptr;
    }

    tensor &operator=(const tensor &other) noexcept {
        if (this != &other) {
            if (x_ != nullptr) dsc_tensor_free(ctx, x_);
            x_ = dsc_new_tensor(ctx, other.x_->n_dim, other.x_->shape, other.x_->dtype);
            memcpy(x_->data, other.x_->data, other.x_->ne * DSC_DTYPE_SIZE[other.x_->dtype]);
        }
        return *this;
    }

    tensor &operator=(tensor &&other) noexcept {
        if (this != &other) {
            if (x_ != nullptr) dsc_tensor_free(ctx, x_);
            x_ = other.x_;
            other.x_ = nullptr;
        }
        return *this;
    }

    ~tensor() noexcept {
        if (x_ != nullptr) dsc_tensor_free(ctx, x_);
    }

    // ============================================================
    // Utilities

    DSC_INLINE int dim(const int idx) const noexcept {
        return x_->shape[dsc_tensor_dim(x_, idx)];
    }

    DSC_INLINE int size() const noexcept {
        return dim(0);
    }

    DSC_INLINE int ndim() const noexcept {
        return x_->n_dim;
    }

    DSC_INLINE T *data() const noexcept {
        return (T *) x_->data;
    }
    // ============================================================
    // Indexing/Slicing

    template<typename ...Args>
        requires (std::is_same_v<Args, int> && ...)
    DSC_INLINE tensor get(Args... indexes) const noexcept {
        constexpr usize n_args = sizeof...(Args);
        static_assert(n_args > 0);

        return dsc_tensor_get_idx(ctx, x_, n_args, indexes...);
    }

    template<typename ...Args>
        requires (std::is_same_v<Args, dsc_slice> && ...)
    DSC_INLINE tensor get(Args... slices) const noexcept {
        constexpr usize n_args = sizeof...(Args);
        static_assert(n_args > 0);

        return dsc_tensor_get_slice(ctx, x_, n_args, slices...);
    }

    template<typename ...Args>
        requires (std::is_same_v<Args, dsc_slice> && ...)
    DSC_INLINE tensor& set(const tensor& other, Args... slices) noexcept {
        constexpr usize n_args = sizeof...(Args);
        static_assert(n_args > 0);

        dsc_tensor_set_slice(ctx, x_, other.x_, n_args, slices...);
        return *this;
    }

    // ============================================================
    // Operators

    DSC_INLINE tensor operator+(const tensor &other) const noexcept {
        return dsc_add(ctx, x_, other.x_);
    }
    DSC_INLINE tensor operator+(const T other) const noexcept {
        return dsc_add(ctx, x_, wrap(other).x_);
    }

    DSC_INLINE tensor operator-(const tensor &other) const noexcept {
        return dsc_sub(ctx, x_, other.x_);
    }
    DSC_INLINE tensor operator-(const T other) const noexcept {
        return dsc_sub(ctx, x_, wrap(other).x_);
    }
    DSC_INLINE friend tensor operator-(T scalar, const tensor& other) noexcept {
        return wrap(scalar) - other;
    }

    DSC_INLINE tensor operator*(const tensor &other) const noexcept {
        return dsc_mul(ctx, x_, other.x_);
    }
    DSC_INLINE tensor operator*(const T other) const noexcept {
        return dsc_mul(ctx, x_, wrap(other).x_);
    }
    friend DSC_INLINE tensor operator*(T scalar, const tensor& other) noexcept {
        return wrap(scalar) * other;
    }

    DSC_INLINE tensor operator/(const tensor &other) const noexcept {
        return dsc_div(ctx, x_, other.x_);
    }
    DSC_INLINE tensor operator/(const T other) const noexcept {
        return dsc_div(ctx, x_, wrap(other).x_);
    }
    DSC_INLINE tensor &operator/=(const tensor &other) noexcept {
        dsc_div(ctx, x_, other.x_, x_);
        return *this;
    }

    // Todo: outside
    DSC_INLINE tensor pow(const real<T> exp) const noexcept {
        return dsc_pow(ctx, x_, wrap(exp).x_);
    }

    // ============================================================
    // Friends

    template<typename U>
    friend DSC_INLINE tensor<U> i0(const tensor<U>& x) noexcept;

    template<typename U>
    friend DSC_INLINE tensor<U> sqrt(const tensor<U>& x) noexcept;

    template<typename U>
    friend DSC_INLINE tensor<U> sinc(const tensor<U>& x) noexcept;

    template<typename U>
    friend DSC_INLINE tensor<U> sum(const tensor<U>& x,
                                    int axis, bool keep_dims) noexcept;

    template<typename U>
    friend DSC_INLINE tensor<U> clip(const tensor<U>& x,
                                     U min, U max) noexcept;

    template<typename U, typename... Args>
        requires (std::is_same_v<Args, int> && ...)
    friend DSC_INLINE tensor<U> transpose(const tensor<U>& x,
                                          Args... axes) noexcept;

    template<typename U, typename... Args>
        requires (std::is_same_v<Args, int> && ...)
    friend DSC_INLINE tensor<U> reshape(const tensor<U>& x,
                                        Args... dimensions) noexcept;

    template<typename U, typename... Args>
        requires (std::is_same_v<Args, tensor<U>> && ...)
    friend DSC_INLINE tensor<U> concat(int axis, Args... tensors) noexcept;

    template<typename U>
    friend DSC_INLINE tensor<U> fft(const tensor<U>& x,
                                    int n, int axis) noexcept;
    template<typename U>
    friend DSC_INLINE tensor<U> ifft(const tensor<U>& x,
                                     int n, int axis) noexcept;

    template<typename U>
    friend DSC_INLINE tensor<U> rfft(const tensor<U>& x,
                                     int n, int axis) noexcept;

    template<typename U>
    friend DSC_INLINE tensor<U> irfft(const tensor<U>& x,
                                      int n, int axis) noexcept;

private:
    static DSC_INLINE tensor wrap(const T scalar) noexcept {
        // Todo: evaluate the possibility of wrapping everything on the stack
        if constexpr (dsc_is_type<T, f32>()) {
            return dsc_wrap_f32(ctx, (f32) scalar);
        } else if constexpr (dsc_is_type<T, f64>()) {
            return dsc_wrap_f64(ctx, (f64) scalar);
        } else if constexpr (dsc_is_type<T, c32>()) {
            return dsc_wrap_c32(ctx, (c32) scalar);
        } else if constexpr (dsc_is_type<T, c64>()) {
            return dsc_wrap_c64(ctx, (c64) scalar);
        } else {
            static_assert("T is not supported");
        }
    }
    dsc_tensor *x_;
};

template<typename T>
static DSC_INLINE tensor<T> arange(const int n) noexcept {
    return dsc_arange(ctx, n, dsc_type_mapping<T>::value);
}

template<typename T>
static DSC_INLINE tensor<T> i0(const tensor<T>& x) noexcept {
    return dsc_i0(ctx, x.x_);
}

template<typename T>
static DSC_INLINE tensor<T> sqrt(const tensor<T>& x) noexcept {
    return dsc_sqrt(ctx, x.x_);
}

template<typename T>
static DSC_INLINE tensor<T> sinc(const tensor<T>& x) noexcept {
    return dsc_sinc(ctx, x.x_);
}

template<typename U>
static DSC_INLINE tensor<U> clip(const tensor<U>& x,
                                 const U min, const U max) noexcept {
    return dsc_clip(ctx, x.x_, nullptr, (f64) min, (f64) max);
}

template<typename T>
static DSC_INLINE tensor<T> sum(const tensor<T>& x,
                                const int axis = -1,
                                const bool keep_dims = true) noexcept {
    return dsc_sum(ctx, x.x_, nullptr, axis, keep_dims);
}

template<typename T, typename... Args>
    requires (std::is_same_v<Args, int> && ...)
static DSC_INLINE tensor<T> transpose(const tensor<T>& x,
                                      Args... axes) noexcept {
    constexpr size n_args = sizeof...(Args);

    if constexpr (n_args == 0) return dsc_transpose(ctx, x.x_, 0);
    return dsc_transpose(ctx, x.x_, n_args, axes...);
}

template<typename T, typename... Args>
    requires (std::is_same_v<Args, int> && ...)
static DSC_INLINE tensor<T> reshape(const tensor<T>& x,
                                    Args... dimensions) noexcept {
    constexpr size n_args = sizeof...(Args);
    static_assert(n_args > 0);

    return dsc_reshape(ctx, x.x_, n_args, dimensions...);
}

template<typename T, typename... Args>
    requires (std::is_same_v<Args, tensor<T>> && ...)
static DSC_INLINE tensor<T> concat(const int axis = 0, Args... tensors) noexcept {
    constexpr size n_args = sizeof...(Args);
    static_assert(n_args > 1);

    return dsc_concat(ctx, axis, n_args, tensors.x_...);
}

template<typename T>
static DSC_INLINE tensor<T> fft(const tensor<T>& x,
                                int n, int axis) noexcept {
    return dsc_fft(ctx, x.x_, nullptr, n, axis);
}

template<typename T>
static DSC_INLINE tensor<T> ifft(const tensor<T>& x,
                                 int n, int axis) noexcept {
    return dsc_ifft(ctx, x.x_, nullptr, n, axis);
}

template<typename T>
static DSC_INLINE tensor<T> rfft(const tensor<T>& x,
                                 int n, int axis) noexcept {
    return dsc_rfft(ctx, x.x_, nullptr, n, axis);
}

template<typename T>
static DSC_INLINE tensor<T> irfft(const tensor<T>& x,
                                  int n, int axis) noexcept {
    return dsc_irfft(ctx, x.x_, nullptr, n, axis);
}
}