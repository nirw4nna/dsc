# Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
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
import time
from utils import WARMUP, BENCH_STEPS, random_nd
import random
import matplotlib.pyplot as plt
from tabulate import tabulate

DTYPES = [np.float32, np.float64, np.complex64, np.complex128]


def plot(np_latency, dsc_latency, unit: str):
    operations = list(np_latency.keys())
    x = np.arange(len(operations))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width / 2, np_latency.values(), width, label='NumPy')
    rects2 = ax.bar(x + width / 2, dsc_latency.values(), width, label='DSC')

    ax.set_ylabel(f'Latency ({unit})')
    ax.set_title('Latency Comparison when X = [60 x 60000]')
    ax.set_xticks(x)
    ax.set_xticklabels(operations, rotation=45, ha='right')
    ax.legend()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()

    plt.show()


def draw_table(np_latency, dsc_latency, unit: str):
    table_data = []
    for op in np_latency.keys():
        table_data.append([op, np_latency[op], dsc_latency[op], dsc_latency[op] / np_latency[op]])

    headers = ['Operation', f'NumPy Latency ({unit})', f'DSC Latency ({unit})', 'Ratio (DSC/NumPy)']
    table = tabulate(table_data, headers=headers, floatfmt=".2f", tablefmt="grid")
    print(table)


def bench(op, *args, **kwargs) -> float:
    if 'out' in kwargs and kwargs['out'] is None:
        del kwargs['out']

    def _call_op():
        op(*args, **kwargs)

    for _ in range(WARMUP):
        _call_op()

    op_time = float('+inf')
    for _ in range(BENCH_STEPS):
        start_ = time.perf_counter()
        _call_op()
        this_time = time.perf_counter() - start_
        op_time = this_time if this_time < op_time else op_time
    return op_time


def bench_binary(show_plot: bool = True):
    ops = {
        'add': (np.add, dsc.add),
        'addc': (np.add, dsc.add),
        'sub': (np.subtract, dsc.sub),
        'subc': (np.subtract, dsc.sub),
        'mul': (np.multiply, dsc.mul),
        'mulc': (np.multiply, dsc.mul),
        'true_div': (np.true_divide, dsc.true_div),
        'true_divc': (np.true_divide, dsc.true_div),
        # For plotting purposes, it's better not to include pow with the other functions as the time it takes to evaluate
        # it is way higher than all the other functions. This will cause the graph not to show the actual variance
        # between the two implementations in a meaningful way for all the other functions.
        # 'pow': (np.power, dsc.power),
        # 'powc': (np.power, dsc.power)
    }
    np_latency = {}
    dsc_latency = {}
    for op_name in ops.keys():
        np_op, dsc_op = ops[op_name]
        is_scalar = op_name.endswith('c')
        for dtype in DTYPES:
            shape = [60, 60_000]
            a = random_nd(shape, dtype)
            if is_scalar:
                if dtype == np.complex64 or dtype == np.complex128:
                    b = complex(random.random(), random.random())
                else:
                    b = random.random()
            else:
                b = random_nd(shape, dtype)
            out = np.empty_like(a)

            a_dsc = dsc.from_numpy(a)
            if is_scalar:
                b_dsc = b
            else:
                b_dsc = dsc.from_numpy(b)
            out_dsc = dsc.from_numpy(out)

            np_latency[f'{op_name}_{dtype.__name__}'] = bench(np_op, a, b, out=out) * 1e3
            dsc_latency[f'{op_name}_{dtype.__name__}'] = bench(dsc_op, a_dsc, b_dsc, out=out_dsc) * 1e3

    draw_table(np_latency, dsc_latency, 'ms')

    if show_plot:
        plot(np_latency, dsc_latency, 'ms')


def bench_unary(show_plot: bool = True):
    ops = {
        'sin': (np.sin, dsc.sin),
        'sinc': (np.sinc, dsc.sinc),
        'cos': (np.cos, dsc.cos),
        'logn': (np.log, dsc.logn),
        'log2': (np.log2, dsc.log2),
        'log10': (np.log10, dsc.log10),
        'exp': (np.exp, dsc.exp),
        'sqrt': (np.sqrt, dsc.sqrt),
        'i0': (np.i0, dsc.i0),
    }
    np_latency = {}
    dsc_latency = {}

    for op_name in ops.keys():
        np_op, dsc_op = ops[op_name]
        for dtype in DTYPES:
            if op_name == 'i0' and (dtype == np.complex64 or dtype == np.complex128):
                continue

            shape = [60, 60_000]
            a = random_nd(shape, dtype)
            out = np.empty_like(a)
            a_dsc = dsc.from_numpy(a)
            out_dsc = dsc.from_numpy(out)

            if op_name in ['sinc', 'i0']:
                # These NumPy functions don't support the out keyword parameter
                out = None
            if op_name in ['i0']:
                # These DSC functions don't support the out keyword parameter
                out_dsc = None

            np_latency[f'{op_name}_{dtype.__name__}'] = bench(np_op, a, out=out) * 1e3
            dsc_latency[f'{op_name}_{dtype.__name__}'] = bench(dsc_op, a_dsc, out=out_dsc) * 1e3

    draw_table(np_latency, dsc_latency, 'ms')

    if show_plot:
        plot(np_latency, dsc_latency, 'ms')


def bench_unary_along_axis(show_plot: bool = True):
    ops = {
        'sum': (np.sum, dsc.sum),
        'mean': (np.mean, dsc.mean),
        'max': (np.max, dsc.max),
        'min': (np.min, dsc.min),
    }
    np_latency = {}
    dsc_latency = {}

    for op_name in ops.keys():
        np_op, dsc_op = ops[op_name]
        for dtype in DTYPES:
            shape = [60, 60_000]
            a = random_nd(shape, dtype)
            out = np.empty((1, 60_000), dtype=dtype)
            a_dsc = dsc.from_numpy(a)
            out_dsc = dsc.from_numpy(out)

            np_latency[f'{op_name}_{dtype.__name__}'] = bench(np_op, a, out=out, axis=0, keepdims=True) * 1e3
            dsc_latency[f'{op_name}_{dtype.__name__}'] = bench(dsc_op, a_dsc, out=out_dsc, axis=0, keepdims=True) * 1e3

    draw_table(np_latency, dsc_latency, 'ms')

    if show_plot:
        plot(np_latency, dsc_latency, 'ms')


if __name__ == '__main__':
    # bench_binary(show_plot=True)
    # bench_unary(show_plot=True)
    bench_unary_along_axis(show_plot=True)
