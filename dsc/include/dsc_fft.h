#pragma once

#include "dsc.h"
#include <cmath>

struct dsc_fft_plan {
    void *twiddles;
    int n;
    dsc_dtype dtype;
};


static DSC_INLINE int dsc_fft_best_n(const int n) noexcept {
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

static DSC_INLINE usize dsc_fft_storage(const int n) noexcept {
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

    return twiddle_storage;
}

extern void dsc_init_plan(dsc_fft_plan *plan, int n,
                          dsc_dtype dtype) noexcept;

template<typename T>
extern void dsc_cfft(dsc_fft_plan *plan,
                     T *DSC_RESTRICT x,
                     T *DSC_RESTRICT out,
                     bool forward) noexcept;

template<typename T>
extern void dsc_rfft(dsc_fft_plan *plan,
                     T *DSC_RESTRICT x,
                     T *DSC_RESTRICT out,
                     bool forward) noexcept;

