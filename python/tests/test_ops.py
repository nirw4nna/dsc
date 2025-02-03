# Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
# All rights reserved.
#
# This code is licensed under the terms of the 3-clause BSD license
# (https://opensource.org/license/bsd-3-clause).
from random import randint

import dsc
import numpy as np
import random
import pytest
from typing import List
import math
from itertools import permutations
import os


DEVICE = os.getenv('DEVICE', default='cpu')


@pytest.fixture(scope='session', autouse=True)
def session_fixture():
    # This is invoked once before starting the test session
    dsc.init(int(2**30))
    dsc.set_default_device(DEVICE)
    yield


@pytest.fixture(autouse=True)
def teardown_fixture():
    # This is invoked automatically after each test
    yield


def all_close(actual: dsc.Tensor, target: np.ndarray, eps=1e-2):
    actual_np = actual.numpy()
    diffs = ~np.isclose(actual_np, target, atol=eps, rtol=eps, equal_nan=True)
    close = len(actual_np[diffs]) == 0
    return close


def random_nd(shape: List[int], dtype: np.dtype = np.float64):
    return np.random.randn(*tuple(shape)).astype(dtype)


DTYPES = [np.float32, np.float64, np.complex64, np.complex128]
DSC_DTYPES = {
    np.float32: dsc.Dtype.F32,
    np.float64: dsc.Dtype.F64,
    np.complex64: dsc.Dtype.C32,
    np.complex128: dsc.Dtype.C64,
}


class TestOps:
    def test_binary(self):
        ops = {
            'add': (np.add, dsc.add, '+'),
            'sub': (np.subtract, dsc.sub, '-'),
            'mul': (np.multiply, dsc.mul, '*'),
            'div': (np.true_divide, dsc.true_div, '/'),
            'power': (np.power, dsc.power, '**'),
        }
        for op_name in ops.keys():
            np_op, dsc_op, symbol = ops[op_name]
            for dtype in DTYPES:
                print(f'Testing operator {op_name} with {dtype.__name__} on {DEVICE}')
                shape = [random.randint(2, 10) for _ in range(4)]
                x = random_nd(shape, dtype=dtype)
                x_dsc = dsc.from_numpy(x)

                # Same shape
                y = random_nd(shape, dtype=dtype)
                y_dsc = dsc.from_numpy(y)

                res_np = np_op(x, y)
                res_dsc = dsc_op(x_dsc, y_dsc)
                r_res_np = eval(f'y {symbol} x')
                r_res_dsc = eval(f'y_dsc {symbol} x_dsc')
                assert all_close(res_dsc, res_np)
                assert all_close(r_res_dsc, r_res_np)

                # Broadcasting
                collapse_idx = random.randint(0, 3)
                shape[collapse_idx] = 1
                y_b = random_nd(shape, dtype=dtype)
                y_dsc_b = dsc.from_numpy(y_b)
                res_np_b = np_op(x, y_b)
                res_dsc_b = dsc_op(x_dsc, y_dsc_b)
                r_res_np_b = eval(f'y_b {symbol} x')
                r_res_dsc_b = eval(f'y_dsc_b {symbol} x_dsc')
                assert all_close(res_dsc_b, res_np_b)
                assert all_close(r_res_dsc_b, r_res_np_b)

                # Scalar
                if dtype == np.complex64 or dtype == np.complex128:
                    y_s = complex(random.random(), random.random())
                else:
                    y_s = random.random()

                res_np_s = np_op(x, y_s)
                res_dsc_s = dsc_op(x_dsc, y_s)
                r_res_np_s = eval(f'y_s {symbol} x')
                r_res_dsc_s = eval(f'y_s {symbol} x_dsc')

                assert all_close(res_dsc_s, res_np_s)
                assert all_close(r_res_dsc_s, r_res_np_s)

    def test_matmul(self):
        def _mnk() -> tuple[int, int, int]:
            return randint(2, 50), randint(2, 50), randint(2, 50)

        def _test_matmul(shape_a: List[int], shape_b: List[int], dt: np.dtype):
            print(f'Testing {shape_a} @ {shape_b} with {dt.__name__} on {DEVICE}')
            xa = random_nd(shape_a, dtype=dt)
            xb = random_nd(shape_b, dtype=dt)
            xa_dsc = dsc.from_numpy(xa)
            xb_dsc = dsc.from_numpy(xb)

            res = xa @ xb
            res_dsc = xa_dsc @ xb_dsc
            assert all_close(res_dsc, res)

        for dtype in DSC_DTYPES:
            if dtype == np.complex64 or dtype == np.complex128:
                continue

            # 2D matrices
            for _ in range(5):
                m, n, k = _mnk()
                _test_matmul([m, k], [k, n], dtype)

            # Batched case
            for _ in range(5):
                batch_1, batch_2 = randint(2, 10), randint(2, 10)
                m, n, k = _mnk()
                _test_matmul([batch_1, batch_2, m, k], [batch_1, batch_2, k, n], dtype)

            # Batched case with broadcasting
            for batch_1 in range(1, 6):
                for batch_2 in range(1, 6):
                    m, n, k = _mnk()
                    _test_matmul([batch_1 if batch_1%2 == 0 else 1,
                                  batch_2 if batch_2%2 == 0 else 1, m, k],
                                 [batch_1 if batch_1%2 == 1 else 1,
                                  batch_2 if batch_2%2 == 1 else 1, k, n],
                                 dtype)

    def test_unary(self):
        ops = {
            'sin': (np.sin, dsc.sin),
            'sinc': (np.sinc, dsc.sinc),
            'cos': (np.cos, dsc.cos),
            'logn': (np.log, dsc.logn),
            'log2': (np.log2, dsc.log2),
            'log10': (np.log10, dsc.log10),
            'exp': (np.exp, dsc.exp),
            'sqrt': (np.sqrt, dsc.sqrt),
            'absolute': (np.absolute, dsc.absolute),
            'angle': (np.angle, dsc.angle),
            'conj': (np.conj, dsc.conj),
            'real': (np.real, dsc.real),
            'imag': (np.imag, dsc.imag),
            'i0': (np.i0, dsc.i0),
        }
        for op_name in ops.keys():
            np_op, dsc_op = ops[op_name]
            for dtype in DTYPES:
                if op_name == 'i0' and (
                    dtype == np.complex64 or dtype == np.complex128
                ):
                    continue

                print(f'Testing {op_name} with {dtype.__name__} on {DEVICE}')
                x = random_nd([random.randint(1, 10) for _ in range(4)], dtype=dtype)
                x_dsc = dsc.from_numpy(x)

                res_np = np_op(x)
                res_dsc = dsc_op(x_dsc)
                assert all_close(res_dsc, res_np)

    def test_clip(self):
        for dtype in DTYPES:
            print(f'Testing clip with {dtype.__name__} on {DEVICE}')
            x = np.arange(10).astype(dtype) - 5
            x_dsc = dsc.from_numpy(x)

            assert all_close(dsc.clip(x_dsc, -2, 2), np.clip(x, -2, 2))
            assert all_close(dsc.clip(x_dsc, -3), np.clip(x, -3, None))
            assert all_close(dsc.clip(x_dsc, None, 2), np.clip(x, None, 2))

    def test_unary_axis(self):
        ops = {
            'sum': (np.sum, dsc.sum),
            'mean': (np.mean, dsc.mean),
            'max': (np.max, dsc.max),
            'min': (np.min, dsc.min),
        }
        for op_name in ops.keys():
            np_op, dsc_op = ops[op_name]
            for dtype in DTYPES:
                for axis in range(-4, 4):
                    print(f'Testing {op_name} with {dtype.__name__} along axis {axis} on {DEVICE}')
                    x = random_nd(
                        [random.randint(1, 10) for _ in range(4)], dtype=dtype
                    )
                    x_dsc = dsc.from_numpy(x)

                    res_np = np_op(x, axis=axis, keepdims=True)
                    res_dsc = dsc_op(x_dsc, axis=axis, keepdims=True)
                    assert all_close(res_dsc, res_np)

                    res_np_2 = np_op(x, axis=axis, keepdims=False)
                    res_dsc_2 = dsc_op(x_dsc, axis=axis, keepdims=False)
                    assert all_close(res_dsc_2, res_np_2)


class TestInit:
    def test_arange(self):
        for _ in range(10):
            n = random.randint(1, 10_000)

            for dtype in DTYPES:
                print(f'Tensing arange with N={n} and dtype={dtype.__name__} on {DEVICE}')
                res_np = np.arange(n, dtype=dtype)
                res_dsc = dsc.arange(n, dtype=DSC_DTYPES[dtype])
                assert all_close(res_dsc, res_np)

    def test_random(self):
        for _ in range(10):
            shape = tuple([random.randint(1, 10) for _ in range(4)])
            for dtype in DTYPES:
                if dtype == np.complex64 or dtype == np.complex128:
                    continue
                print(f'Tensing randn with dtype={dtype.__name__} on {DEVICE}')

                res_np = np.random.randn(*shape).astype(dtype)
                res_dsc = dsc.randn(*shape, dtype=DSC_DTYPES[dtype])
                res_dsc_np = res_dsc.numpy()

                assert res_dsc_np.dtype == res_np.dtype
                assert res_dsc_np.shape == res_np.shape


class TestIndexing:
    def test_get_idx(self):
        # The idea is to start with 1D tensors and then, for all dtypes, test with a growing number of indexes
        # from 1 up to the number of dimensions (to select a scalar value). Given the number of indexes we generate
        # a bunch of random pairs to try and cover most use cases.
        for n_dim in range(4):
            for dtype in DTYPES:
                x = random_nd([10 for _ in range(n_dim + 1)], dtype=dtype)
                x_dsc = dsc.from_numpy(x)

                for indexes in range(n_dim + 1):
                    for _ in range(10):
                        idx = tuple(random.randint(-10, 9) for _ in range(indexes + 1))
                        res = x[idx]
                        res_dsc = x_dsc[idx]
                        if isinstance(res_dsc, dsc.Tensor):
                            assert all_close(res_dsc, res)
                        else:
                            assert np.isclose(res, res_dsc)

    @staticmethod
    def _validate_slice(sl: slice, max_dim: int) -> bool:
        s_start = sl.start
        s_stop = sl.stop
        s_step = sl.step
        san_start = s_start if s_start >= 0 else s_start + max_dim
        san_stop = s_stop if s_stop >= 0 else s_stop + max_dim
        # Some of these checks should probably be handles gracefully by DSC
        if s_step == 0 or san_start == san_stop:
            return False
        if (s_step > 0 and san_stop < san_start) or (
            s_step < 0 and san_stop > san_start
        ):
            return False
        return True

    def test_get_slice(self):
        # Note: this should probably be more exhaustive
        x_1d = random_nd([10], np.float32)
        x_1d_dsc = dsc.from_numpy(x_1d)

        for start in range(-10, 10):
            for stop in range(-10, 10):
                for step in range(-10, 10):
                    s = slice(start, stop, step)
                    if not TestIndexing._validate_slice(s, 10):
                        continue
                    assert all_close(x_1d_dsc[s], x_1d[s])

        x_2d = random_nd([5, 5], np.float32)
        x_2d_dsc = dsc.from_numpy(x_2d)
        for start in range(-5, 5):
            for stop in range(-5, 5):
                for step in range(-5, 5):
                    s = slice(start, stop, step)
                    if not TestIndexing._validate_slice(s, 5):
                        continue
                    assert all_close(
                        x_2d_dsc[(slice(None, None, None), s)],
                        x_2d[(slice(None, None, None), s)],
                    )

        for extra_dim in range(-5, 5):
            for start in range(-5, 5):
                for stop in range(-5, 5):
                    for step in range(-5, 5):
                        s = slice(start, stop, step)
                        if not TestIndexing._validate_slice(s, 5):
                            continue

                        x_dsc_1 = x_2d_dsc[(extra_dim, s)]
                        x_np_1 = x_2d[(extra_dim, s)]
                        assert all_close(x_dsc_1, x_np_1)

                        x_dsc_2 = x_2d_dsc[(s, extra_dim)]
                        x_np_2 = x_2d[(s, extra_dim)]
                        assert all_close(x_dsc_2, x_np_2)

    def test_set_idx(self):
        for n_dim in range(1, 5):
            for dtype in DTYPES:
                x = random_nd([10 for _ in range(n_dim)], dtype=dtype)
                x_dsc = dsc.from_numpy(x)

                for indexes in range(1, n_dim):
                    for _ in range(10):
                        idx = tuple(random.randint(-10, 9) for _ in range(indexes))
                        val = (
                            random.random() + 1
                            if indexes == n_dim
                            else random_nd(
                                [10 for _ in range(n_dim - indexes)], dtype=dtype
                            )
                        )
                        x[idx] = val
                        x_dsc[idx] = val
                        assert all_close(x_dsc, x)

    def test_set_slice(self):
        def _shape_from_slice(sl: slice, max_dim: int) -> List[int]:
            real_start = sl.start if sl.start >= 0 else sl.start + max_dim
            real_stop = sl.stop if sl.stop >= 0 else sl.stop + max_dim
            return [math.ceil(math.fabs(real_start - real_stop) / math.fabs(sl.step))]

        # This is not exhaustive, but it's good enough for now
        x_1d = random_nd([10], np.float32)
        x_1d_dsc = dsc.from_numpy(x_1d)

        x_1d[:] = np.ones(10, dtype=np.float32)
        x_1d_dsc[:] = np.ones(10, dtype=np.float32)
        assert all_close(x_1d_dsc, x_1d)

        for start in range(-10, 10):
            for stop in range(-10, 10):
                for step in range(-10, 10):
                    s = slice(start, stop, step)
                    if not TestIndexing._validate_slice(s, 10):
                        continue
                    x_1d[s] = 1516.0
                    x_1d_dsc[s] = 1516.0
                    assert all_close(x_1d_dsc, x_1d)

                    val_shape = _shape_from_slice(s, 10)
                    val = random_nd(val_shape, dtype=np.float32)
                    x_1d[s] = val
                    x_1d_dsc[s] = val
                    assert all_close(x_1d_dsc, x_1d)

        x_2d = random_nd([5, 5], np.float32)
        x_2d_dsc = dsc.from_numpy(x_2d)

        for extra_dim in range(-5, 5):
            for start in range(-5, 5):
                for stop in range(-5, 5):
                    for step in range(-5, 5):
                        s = slice(start, stop, step)
                        if not TestIndexing._validate_slice(s, 5):
                            continue

                        x_2d[(extra_dim, s)] = 12.0
                        x_2d_dsc[(extra_dim, s)] = 12.0
                        assert all_close(x_2d_dsc, x_2d)

                        x_2d[(s, extra_dim)] = -1.55
                        x_2d_dsc[(s, extra_dim)] = -1.55
                        assert all_close(x_2d_dsc, x_2d)

                        val_shape = _shape_from_slice(s, 5)
                        val = random_nd(val_shape, np.float32)
                        x_2d[(extra_dim, s)] = val
                        x_2d_dsc[(extra_dim, s)] = val
                        assert all_close(x_2d_dsc, x_2d)

                        val = random_nd(val_shape, np.float32)
                        x_2d[(s, extra_dim)] = val
                        x_2d_dsc[(s, extra_dim)] = val
                        assert all_close(x_2d_dsc, x_2d)


def test_creation():
    for n_dim in range(4):
        for dtype in DTYPES:
            shape = tuple(random.randint(1, 20) for _ in range(n_dim + 1))
            fill = random.random()
            if dtype == np.complex64 or dtype == np.complex128:
                fill = complex(random.random(), random.random())
            x = np.full(shape, fill_value=fill, dtype=dtype)
            x_dsc = dsc.full(shape, fill_value=fill, dtype=DSC_DTYPES[dtype])
            assert all_close(x_dsc, x)

            like = np.ones([random.randint(1, 10) for _ in range(n_dim + 1)])

            x = np.full_like(like, fill_value=fill, dtype=dtype)
            x_dsc = dsc.full_like(like, fill_value=fill, dtype=DSC_DTYPES[dtype])
            assert all_close(x_dsc, x)

            x = np.ones(shape, dtype=dtype)
            x_dsc = dsc.ones(shape, dtype=DSC_DTYPES[dtype])
            assert all_close(x_dsc, x)

            x = np.ones_like(like, dtype=dtype)
            x_dsc = dsc.ones_like(like, dtype=DSC_DTYPES[dtype])
            assert all_close(x_dsc, x)

            x = np.zeros(shape, dtype=dtype)
            x_dsc = dsc.zeros(shape, dtype=DSC_DTYPES[dtype])
            assert all_close(x_dsc, x)

            x = np.zeros_like(like, dtype=dtype)
            x_dsc = dsc.zeros_like(like, dtype=DSC_DTYPES[dtype])
            assert all_close(x_dsc, x)


def test_reshape():
    x = np.ones((10, 10))
    x_dsc = dsc.from_numpy(x)
    assert all_close(x_dsc.reshape(4, 5, 5), x.reshape(4, 5, 5))
    assert all_close(x_dsc.reshape([4, 5, 5]), x.reshape([4, 5, 5]))
    assert all_close(x_dsc.reshape((4, 5, 5)), x.reshape((4, 5, 5)))

    assert all_close(x_dsc.reshape(-1, 5), x.reshape(-1, 5))
    assert all_close(x_dsc.reshape([-1, 5]), x.reshape([-1, 5]))
    assert all_close(x_dsc.reshape((-1, 5)), x.reshape((-1, 5)))


def test_concat():
    for n_dim in range(1, 5):
        for dtype in DTYPES:
            shape = [random.randint(2, 10) for _ in range(n_dim)]
            for axis_idx in range(n_dim):
                print(
                    f'Testing concat with {n_dim}-dimensional tensors of type {dtype.__name__} on axis {axis_idx}'
                )
                shape_x1 = list(shape)
                shape_x1[axis_idx] = random.randint(2, 10)
                shape_x2 = list(shape)
                shape_x2[axis_idx] = random.randint(2, 10)
                x1 = random_nd(shape_x1, dtype)
                x2 = random_nd(shape_x2, dtype)
                x1_dsc = dsc.from_numpy(x1)
                x2_dsc = dsc.from_numpy(x2)

                res_np = np.concat((x1, x2), axis_idx)
                res_dsc = dsc.concat((x1_dsc, x2_dsc), axis_idx)
                assert all_close(res_dsc, res_np)

                # Test flatten
                res_np_flat = np.concat((x1, x2), None)
                res_dsc_flat = dsc.concat((x1_dsc, x2_dsc), None)
                assert all_close(res_dsc_flat, res_np_flat)


def test_transpose():
    for n_dim in range(1, 5):
        for dtype in DTYPES:
            print(
                f'Testing transpose with {n_dim}-dimensional tensors of type {dtype.__name__}'
            )
            shape = [random.randint(2, 10) for _ in range(n_dim)]
            x = random_nd(shape, dtype)
            x_dsc = dsc.from_numpy(x)
            # Simple transpose
            res_np_simple = np.transpose(x)
            res_dsc_simple = dsc.transpose(x_dsc)
            assert all_close(res_dsc_simple, res_np_simple)

            # Test with all the permutations of axes
            for axes in permutations(range(n_dim)):
                res_np = np.transpose(x, axes)
                res_dsc = dsc.transpose(x_dsc, axes)
                assert all_close(res_dsc, res_np)
