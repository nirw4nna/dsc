import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import python.dsc as dsc
import time
from typing import List
import numpy as np
import matplotlib.pyplot as plt


WARMUP = 2
BENCH_STEPS = 5


def random_nd(shape: List[int], dtype: np.dtype = np.float64):
    return np.random.randn(*tuple(shape)).astype(dtype)


def bench_fft(plot: bool = True):
    dsc.init(1024 * 1024 * 1024 * 2)
    # MFLOPS = 5N * log2(N) / us (https://www.fftw.org/speed/method.html)
    np_f32_flops = {}
    np_f64_flops = {}
    dsc_f32_flops = {}
    dsc_f64_flops = {}

    def _bench(x, fft_op):
        for _ in range(WARMUP):
            fft_op(x)

        fft_time = float('+inf')
        for _ in range(BENCH_STEPS):
            start_ = time.perf_counter()
            fft_op(x)
            this_time = (time.perf_counter() - start_) * 1e9
            fft_time = this_time if this_time < fft_time else fft_time
        return fft_time

    for n_ in range(4, 20):
        n = 2**n_
        flops = 5 * n * n_
        x_f32 = random_nd([n], dtype=np.float32)
        x_f64 = random_nd([n], dtype=np.float64)
        x_dsc_f32 = dsc.from_numpy(x_f32)

        np_f32_flops[n] = flops / _bench(x_f32, np.fft.fft)
        np_f64_flops[n] = flops / _bench(x_f64, np.fft.fft)
        dsc_f32_flops[n] = flops / _bench(x_dsc_f32, dsc.fft)
        dsc.clear()
        x_dsc_f64 = dsc.from_numpy(x_f64)
        dsc_f64_flops[n] = flops / _bench(x_dsc_f64, dsc.fft)
        dsc.clear()

    if plot:
        labels = list(np_f32_flops.keys())
        x = range(len(labels))
        plt.plot(x, list(np_f32_flops.values()), marker='d', label='NumPy (f32)')
        plt.plot(x, list(np_f64_flops.values()), marker='s', label='NumPy (f64)')
        plt.plot(x, list(dsc_f32_flops.values()),  marker='v', label='DSC (f32)')
        plt.plot(x, list(dsc_f64_flops.values()),  marker='^', label='DSC (f64)')
        plt.grid(True)
        plt.xlabel('Size')
        plt.ylabel('GFLOPS')
        plt.title('1D FFT')
        plt.xticks(x, labels, rotation=90)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    bench_fft()
