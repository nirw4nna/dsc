#include "dsc.h"
#include "pocketfft.h"
#include <cstring>

#define N 64


static void fill_random(f64 *x, const int n) noexcept {
    for (int i = 0; i < n; ++i)
        x[i] = (f64) (rand() / (RAND_MAX + 1.0) - 0.5);
}

static f64 max_diff(const f64 *A, const f64 *B, const int n) noexcept {
    static constexpr f64 inf = -std::numeric_limits<f64>::infinity();

    f64 diff = inf;
    for (int i = 0; i < n; ++i) {
        const f64 this_diff = std::abs(A[i] - B[i]);
        if (this_diff > diff)
            diff = this_diff;
    }

    return diff;
}

//struct my_plan {
//    f64 *__restrict twiddles;
//    int n;
//};
//
//my_plan *my_create_plan(const int n) noexcept {
//    my_plan *plan = (my_plan *) malloc(sizeof(my_plan));
//    plan->n = n;
//
//    // Precompute twiddles
//    u64 twiddle_storage = 0;
//    for (int twiddle_n = 2; twiddle_n <= n; twiddle_n <<= 1) {
//        // Given N we need N/2 twiddle factors but each twiddle factor has a real and an imaginary part so that's N
//        twiddle_storage += twiddle_n;
//    }
//
//    plan->twiddles = (f64 *) malloc(twiddle_storage * sizeof(f64));
//
//    for (int twiddle_n = 2; twiddle_n <= n; twiddle_n <<= 1) {
//        const int twiddle_n2 = twiddle_n >> 1;
//        for (int k = 0; k < twiddle_n2; ++k) {
//            const f64 theta = (-2. * M_PI * k) / (f64) (twiddle_n);
//            plan->twiddles[(2 * (twiddle_n2 - 1)) + (2 * k)] = cos(theta);
//            plan->twiddles[(2 * (twiddle_n2 - 1)) + (2 * k) + 1] = sin(theta);
//        }
//    }
//
//    return plan;
//}
//
//void my_destroy_plan(my_plan *plan) noexcept {
//    free(plan->twiddles);
//    free(plan);
//}
//
//static void my_fft(c64 *__restrict x,
//                   c64 *__restrict work,
//                   const f64 *__restrict twiddles,
//                   const int n,
//                   const f64 sign) noexcept {
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
//    my_fft(work, x, twiddles, n2, sign);
//    // FFT of odd indexes
//    my_fft(work + n2, x, twiddles, n2, sign);
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
//
//void my_fft(my_plan *plan, c64 *__restrict x, c64 *__restrict work) noexcept {
//    my_fft(x, work, plan->twiddles, plan->n, 1);
//}

int main() {
    dsc_ctx *ctx = dsc_ctx_init(1024 * 1024);
    dsc_plan_fft(ctx, N);

    dsc_tensor *x = dsc_tensor_1d(ctx, C64, N);

    cfft_plan pocket_plan = make_cfft_plan(N);

    c64 x_pocket[N];
    c64 my_x[N];

    fill_random((f64 *) x_pocket, 2 * N);

    memcpy(x->data, x_pocket, N * sizeof(c64));
    memcpy(my_x, x_pocket, N * sizeof(c64));

    dsc_tensor *x_fft;
    {
        x_fft = dsc_fft(ctx, x);
    }

    {
        cfft_forward(pocket_plan, (f64 *) x_pocket, 1.);
    }

    const f64 diff = max_diff((f64 *) x_fft->data, (f64 *) x_pocket, 2 * N);
    printf("diff= %.2e\n", diff);

    destroy_cfft_plan(pocket_plan);
    dsc_ctx_free(ctx);
}