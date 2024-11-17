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