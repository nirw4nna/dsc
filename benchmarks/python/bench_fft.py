# Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
# All rights reserved.
#
# This code is licensed under the terms of the 3-clause BSD license
# (https://opensource.org/license/bsd-3-clause).

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import dsc
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import WARMUP, BENCH_STEPS, random_nd


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


def bench_fft(show_plot: bool = True):
    # FLOPS = 5N * log2(N) / s for the FFT (https://www.fftw.org/speed/method.html)

    np_f64_fft_flops = {}
    dsc_f32_fft_flops = {}
    dsc_f64_fft_flops = {}
    np_f64_rfft_flops = {}
    dsc_f32_rfft_flops = {}
    dsc_f64_rfft_flops = {}

    for n_ in range(6, 21):
        n = 2**n_
        flops = 5 * n * n_
        x_f32 = random_nd([n], dtype=np.float32)
        x_f64 = random_nd([n], dtype=np.float64)

        # FFT
        x_dsc_f32 = dsc.from_numpy(x_f32)
        np_f64_fft_flops[n] = flops / bench(x_f64, np.fft.fft)
        dsc_f32_fft_flops[n] = flops / bench(x_dsc_f32, dsc.fft)

        x_dsc_f64 = dsc.from_numpy(x_f64)
        dsc_f64_fft_flops[n] = flops / bench(x_dsc_f64, dsc.fft)

        del x_dsc_f32
        del x_dsc_f64

        # RFFT
        flops = 0.5 * flops
        x_dsc_f32 = dsc.from_numpy(x_f32)
        np_f64_rfft_flops[n] = flops / bench(x_f64, np.fft.rfft)
        dsc_f32_rfft_flops[n] = flops / bench(x_dsc_f32, dsc.rfft)
        x_dsc_f64 = dsc.from_numpy(x_f64)
        dsc_f64_rfft_flops[n] = flops / bench(x_dsc_f64, dsc.rfft)

    runs = len(np_f64_fft_flops)
    np_f64_fft_mean_flops = sum(np_f64_fft_flops.values()) / runs
    dsc_f64_fft_mean_flops = sum(dsc_f64_fft_flops.values()) / runs
    np_f64_rfft_mean_flops = sum(np_f64_rfft_flops.values()) / runs
    dsc_f64_rfft_mean_flops = sum(dsc_f64_rfft_flops.values()) / runs

    np_speed_fft_f64 = np_f64_fft_mean_flops / dsc_f64_fft_mean_flops
    np_speed_rfft_f64 = np_f64_rfft_mean_flops / dsc_f64_rfft_mean_flops
    print(f'FFT NumPy is {round(np_speed_fft_f64, 1)}X faster (NumPy Avg. GFLOPS={round(np_f64_fft_mean_flops, 2)}'
          f'\tDSC Avg. GFLOPS={round(dsc_f64_fft_mean_flops, 2)})')
    print(f'RFFT NumPy is {round(np_speed_rfft_f64, 1)}X faster (NumPy Avg. GFLOPS={round(np_f64_rfft_mean_flops, 2)}'
          f'\tDSC Avg. GFLOPS={round(dsc_f64_rfft_mean_flops, 2)})')

    if show_plot:
        labels = list(np_f64_fft_flops.keys())
        x = range(len(labels))
        plt.plot(x, list(np_f64_fft_flops.values()), marker='s', label='NumPy FFT (f64)')
        plt.plot(x, list(dsc_f32_fft_flops.values()),  marker='v', label='DSC FFT (f32)')
        plt.plot(x, list(dsc_f64_fft_flops.values()),  marker='^', label='DSC FFT (f64)')
        plt.plot(x, list(np_f64_rfft_flops.values()), marker='D', label='NumPy RFFT (f64)')
        plt.plot(x, list(dsc_f32_rfft_flops.values()),  marker='<', label='DSC RFFT (f32)')
        plt.plot(x, list(dsc_f64_rfft_flops.values()),  marker='>', label='DSC RFFT (f64)')
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
    bench_fft(show_plot=True)
