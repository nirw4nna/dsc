import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import python.dsc as dsc
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List


WARMUP = 2
BENCH_STEPS = 5


def bench(x, op):
    for _ in range(WARMUP):
        op(x)

    op_time_ns = float('+inf')
    for _ in range(BENCH_STEPS):
        start_ = time.perf_counter()
        op(x)
        this_time = (time.perf_counter() - start_) * 1e9
        op_time_ns = this_time if this_time < op_time_ns else op_time_ns
    return op_time_ns


def random_nd(shape: List[int], dtype: np.dtype = np.float64):
    return np.random.randn(*tuple(shape)).astype(dtype)


def bench_fft(plot: bool = True):
    dsc.init(1024 * 1024 * 1024 * 2)
    # FLOPS = 5N * log2(N) / s (https://www.fftw.org/speed/method.html)
    np_f32_flops = {}
    np_f64_flops = {}
    dsc_f32_flops = {}
    dsc_f64_flops = {}

    for n_ in range(6, 21):
        n = 2**n_
        flops = 5 * n * n_
        x_f32 = random_nd([n], dtype=np.float32)
        x_f64 = random_nd([n], dtype=np.float64)
        x_dsc_f32 = dsc.from_numpy(x_f32)

        np_f32_flops[n] = flops / bench(x_f32, np.fft.fft)
        np_f64_flops[n] = flops / bench(x_f64, np.fft.fft)
        dsc_f32_flops[n] = flops / bench(x_dsc_f32, dsc.fft)
        dsc.clear()
        x_dsc_f64 = dsc.from_numpy(x_f64)
        dsc_f64_flops[n] = flops / bench(x_dsc_f64, dsc.fft)
        dsc.clear()

    runs = len(np_f32_flops)
    np_f32_mean_flops = sum(np_f32_flops.values()) / runs
    np_f64_mean_flops = sum(np_f64_flops.values()) / runs
    dsc_f32_mean_flops = sum(dsc_f32_flops.values()) / runs
    dsc_f64_mean_flops = sum(dsc_f64_flops.values()) / runs

    np_speed_f32 = np_f32_mean_flops / dsc_f32_mean_flops
    np_speed_f64 = np_f64_mean_flops / dsc_f64_mean_flops
    print(f'F32: NumPy is {round(np_speed_f32, 1)}X faster (NumPy Avg. GFLOPS={round(np_f32_mean_flops, 2)}\tDSC Avg. '
          f'GFLOPS={round(dsc_f32_mean_flops, 2)})')
    print(f'F64: NumPy is {round(np_speed_f64, 1)}X faster (NumPy Avg. GFLOPS={round(np_f64_mean_flops, 2)}\tDSC Avg. '
          f'GFLOPS={round(dsc_f64_mean_flops, 2)})')

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

        y_min, y_max = plt.ylim()
        y_ticks = np.arange(np.floor(y_min), np.ceil(y_max), 0.5)
        plt.yticks(y_ticks)
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    bench_fft()
