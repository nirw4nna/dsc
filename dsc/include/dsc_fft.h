#pragma once

#include "dsc.h"
#include "dsc_ops.h"
#include <cmath>


struct dsc_fft_plan {
    void *twiddles;
    int n;
    dsc_dtype dtype;
};

namespace {
template<typename T>
void dsc_init_plan(dsc_fft_plan *plan, const int n,
                   const dsc_dtype dtype) {
    static_assert(dsc_is_real<T>(), "Twiddles dtype must be real!");

    T *twiddles = (T *) plan->twiddles;

    for (int twiddle_n = 2; twiddle_n <= n; twiddle_n <<= 1) {
        const int twiddle_n2 = twiddle_n >> 1;
        for (int k = 0; k < twiddle_n2; ++k) {
            const T theta = ((T) (-2.) * dsc_pi<T>() * k) / (T) (twiddle_n);
            twiddles[(2 * (twiddle_n2 - 1)) + (2 * k)] = cos_op()(theta);
            twiddles[(2 * (twiddle_n2 - 1)) + (2 * k) + 1] = sin_op()(theta);
        }
    }

    plan->n = n;
    plan->dtype = dtype;
}

// Todo: profile NOINLINE vs INLINE
template<typename T, dsc_real<T> sign>
DSC_NOINLINE void dsc_fft_pass2(T *DSC_RESTRICT x,
                                T *DSC_RESTRICT work,
                                const dsc_real<T> *DSC_RESTRICT twiddles,
                                const int n) noexcept {
    // Base case
    if (n <= 1) return;

    const int n2 = n >> 1;

    // Divide
    for (int i = 0; i < n2; i++) {
        work[i] = x[2 * i];
        work[i + n2] = x[2 * i + 1];
    }

    // FFT of even indexes
    dsc_fft_pass2<T, sign>(work, x, twiddles, n2);
    // FFT of odd indexes
    dsc_fft_pass2<T, sign>(work + n2, x, twiddles, n2);

    const int t_base = (n2 - 1) << 1;

    // Conquer
    for (int k = 0; k < n2; ++k) {
        T tmp{};
        // Twiddle[k] = [Re, Imag] = [cos(theta), sin(theta)]
        // but cos(theta) = cos(-theta) and sin(theta) = -sin(theta)
        const dsc_real<T> twiddle_r = twiddles[t_base + (2 * k)];
        const dsc_real<T> twiddle_i = sign * twiddles[t_base + (2 * k) + 1];

        // t = w * x_odd[k]
        // x[k] = x_even[k] + t
        // x[k + n/2] = x_even[k] - t

        const T x_odd_k = work[k + n2];
        const T x_even_k = work[k];

        tmp.real = twiddle_r * x_odd_k.real - twiddle_i * x_odd_k.imag;
        tmp.imag = twiddle_r * x_odd_k.imag + twiddle_i * x_odd_k.real;

        x[k].real = x_even_k.real + tmp.real;
        x[k].imag = x_even_k.imag + tmp.imag;
        x[k + n2].real = x_even_k.real - tmp.real;
        x[k + n2].imag = x_even_k.imag - tmp.imag;
    }
}
}

static DSC_STRICTLY_PURE int dsc_fft_best_n(const int n) noexcept {
    // Compute the best fitting N based on the available algorithms.
    // For now, return the power-of-2 closest to n.
    DSC_ASSERT(n > 0);
    // Compute the next power-of-2 for a 32bit integer
    int next_pow2_n = n - 1;
    next_pow2_n |= (next_pow2_n >> 1);
    next_pow2_n |= (next_pow2_n >> 2);
    next_pow2_n |= (next_pow2_n >> 4);
    next_pow2_n |= (next_pow2_n >> 8);
    next_pow2_n |= (next_pow2_n >> 16);
    return next_pow2_n + 1;
}

static DSC_STRICTLY_PURE usize dsc_fft_storage(const int n, const dsc_dtype dtype) noexcept {
    // Compute how much storage is needed for the plan (twiddles * dtype).
    DSC_ASSERT(n > 0);
    // N must be a power-of-2
    DSC_ASSERT((n & (n - 1)) == 0);

    // Todo: a lot of storage can be saved by exploiting the periodic properties of sin/cos
    usize twiddle_storage = 0;
    for (int twiddle_n = 2; twiddle_n <= n; twiddle_n <<= 1) {
        // Given N we need N/2 twiddle factors but each twiddle factor has a real and an imaginary part so that's N
        twiddle_storage += twiddle_n;
    }
    usize dtype_size;
    switch (dtype) {
        case C32:
        case F32:
            dtype_size = DSC_DTYPE_SIZE[F32];
            break;
        case C64:
        case F64:
            dtype_size = DSC_DTYPE_SIZE[F64];
            break;
        default:
            DSC_LOG_ERR("unknown dtype=%d", dtype);
    }
    return twiddle_storage * dtype_size;
}

static void dsc_init_plan(dsc_fft_plan *plan, int n,
                          dsc_dtype dtype) noexcept {
    DSC_ASSERT(n > 0);
    DSC_ASSERT((n & (n - 1)) == 0);

    switch (dtype) {
        case C32:
        case F32:
            dsc_init_plan<f32>(plan, n, F32);
            break;
        case C64:
        case F64:
            dsc_init_plan<f64>(plan, n, F64);
            break;
        default:
            DSC_LOG_ERR("unknown dtype=%d", dtype);
    }
}

template<typename T, bool forward>
static void dsc_cfft(dsc_fft_plan *plan,
                     T *DSC_RESTRICT x,
                     T *DSC_RESTRICT work) noexcept {
    static_assert(dsc_is_complex<T>(), "T must be a complex type, use rfft for real-valued FFTs");

    dsc_fft_pass2<T, forward ? (dsc_real<T>) +1 : (dsc_real<T>) -1>(x, work, (dsc_real<T> *) plan->twiddles, plan->n);

    if constexpr (!forward) {
        // Divide all the elements by N
        const dsc_real<T> scale = (dsc_real<T>) 1 / (dsc_real<T>) plan->n;
        for (int i = 0; i < plan->n; ++i) {
            x[i].real *= scale;
            x[i].imag *= scale;
        }
    }
}

//template<typename T, bool forward>
//extern void dsc_rfft(dsc_fft_plan *plan,
//                     T *DSC_RESTRICT x,
//                     T *DSC_RESTRICT work) noexcept;
//


