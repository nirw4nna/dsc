#include "dsc_fft.h"
#include "dsc_ops.h"

template<typename T>
static consteval T pi() noexcept {
    if constexpr (dsc_is_type<T, f32>()) {
        return M_PIf;
    } else {
        return M_PI;
    }
}

template<typename T>
static void dsc_init_plan(dsc_fft_plan *plan, const int n,
                          const dsc_dtype dtype) {
    static_assert(dsc_is_real<T>(), "FFT twiddles dtype must be real");

    T *twiddles = (T *) plan->twiddles;

    for (int twiddle_n = 2; twiddle_n <= n; twiddle_n <<= 1) {
        const int twiddle_n2 = twiddle_n >> 1;
        for (int k = 0; k < twiddle_n2; ++k) {
            const T theta = ((T) (-2.) * pi<T>() * k) / (T) (twiddle_n);
            twiddles[(2 * (twiddle_n2 - 1)) + (2 * k)] = cos_op()(theta);
            twiddles[(2 * (twiddle_n2 - 1)) + (2 * k) + 1] = sin_op()(theta);
        }
    }

    plan->n = n;
    plan->dtype = dtype;
}

void dsc_init_plan(dsc_fft_plan *plan, const int n,
                   const dsc_dtype dtype) noexcept {
    DSC_ASSERT(dtype == dsc_dtype::F32 || dtype == dsc_dtype::F64);
    DSC_ASSERT(n > 0);
    DSC_ASSERT((n & (n - 1)) == 0);

    switch (dtype) {
        case F32:
            dsc_init_plan<f32>(plan, n, dtype);
            break;
        case F64:
            dsc_init_plan<f64>(plan, n, dtype);
            break;
        default:
            DSC_LOG_FATAL("unsupported dtype=%d", dtype);
    }
}
//static DSC_NOINLINE void dsc_fft_r2(c64 *DSC_RESTRICT x,
//                                    c64 *DSC_RESTRICT work,
//                                    const f64 *DSC_RESTRICT twiddles,
//                                    const int n,
//                                    const f64 sign) noexcept {
//    // Base case
//    if (n <= 1) return;
//
//    const int n2 = n >> 1;
//
//    // Divide
//    for (int i = 0; i < n2; i++) {
//        work[i] = x[2 * i];
//        work[i + n2] = x[2 * i + 1];
//    }
//
//    // FFT of even indexes
//    dsc_fft_r2(work, x, twiddles, n2, sign);
//    // FFT of odd indexes
//    dsc_fft_r2(work + n2, x, twiddles, n2, sign);
//
//    const int t_base = (n2 - 1) << 1;
//
//    // Conquer
//    for (int k = 0; k < n2; ++k) {
//        c64 tmp{};
//        // Twiddle[k] = [Re, Imag] = [cos(theta), sin(theta)]
//        // but cos(theta) = cos(-theta) and sin(theta) = -sin(theta)
//        const f64 twiddle_r = twiddles[t_base + (2 * k)];
//        const f64 twiddle_i = sign * twiddles[t_base + (2 * k) + 1];
//        // t = w * x_odd[k]
//        // x[k] = x_even[k] + t
//        // x[k + n/2] = x_even[k] - t
//
//        const c64 x_odd_k = work[k + n2];
//        const c64 x_even_k = work[k];
//
//        tmp.real = twiddle_r * x_odd_k.real - twiddle_i * x_odd_k.imag;
//        tmp.imag = twiddle_r * x_odd_k.imag + twiddle_i * x_odd_k.real;
//
//        x[k].real = x_even_k.real + tmp.real;
//        x[k].imag = x_even_k.imag + tmp.imag;
//        x[k + n2].real = x_even_k.real - tmp.real;
//        x[k + n2].imag = x_even_k.imag - tmp.imag;
//    }
//}




template<typename T>
void dsc_cfft(dsc_fft_plan *plan,
              T *DSC_RESTRICT x,
              T *DSC_RESTRICT out,
              const bool forward) noexcept {
    static_assert(dsc_is_complex<T>(), "T must be a complex type, use rfft for real-valued FFTs");


}