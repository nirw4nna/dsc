# Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
# All rights reserved.
#
# This code is licensed under the terms of the 3-clause BSD license
# (https://opensource.org/license/bsd-3-clause).

import dsc
import numpy as np
import pytest
from random import randint, random
from typing import List
import math
from itertools import permutations


@pytest.fixture(scope='session', autouse=True)
def session_fixture():
    # This is invoked once before starting the test session
    dsc.init(int(2**30))
    yield


@pytest.fixture(autouse=True)
def teardown_fixture():
    # This is invoked automatically after each test
    yield


def all_close(actual: dsc.Tensor, target: np.ndarray, eps=1e-5):
    actual_np = actual.numpy()
    diffs = ~np.isclose(actual_np, target, atol=eps, rtol=eps, equal_nan=True)
    close = len(actual_np[diffs]) == 0
    return close


def random_nd(shape: List[int], dtype: np.dtype = np.float64):
    if dtype == np.bool:
        return np.random.randint(0, 2, size=tuple(shape)).astype(dtype)
    elif dtype == np.int32:
        # Return a positive integer tensor if the dtype is int32 so that we don't have issues
        # with power
        return np.random.randint(0, 10, size=tuple(shape)).astype(dtype)
    else:
        return np.random.randn(*tuple(shape)).astype(dtype)


DTYPES = [np.bool, np.int32, np.float32, np.float64]
DSC_DTYPES = {
    np.bool: dsc.Dtype.BOOL,
    np.int32: dsc.Dtype.I32,
    np.float32: dsc.Dtype.F32,
    np.float64: dsc.Dtype.F64,
}

def is_float(dtype) -> bool:
    return dtype == np.float32 or dtype == np.float64

def is_bool(dtype) -> bool:
    return dtype == np.bool

def is_integer(dtype) -> bool:
    return dtype == np.int32


class TestOps:
    def test_binary(self):
        ops = {
            'add': (np.add, dsc.add),
            'sub': (np.subtract, dsc.sub),
            'mul': (np.multiply, dsc.mul),
            'div': (np.true_divide, dsc.true_div),
            'power': (np.power, dsc.power),
            'equal': (np.equal, dsc.equal),
            'not_equal': (np.not_equal, dsc.not_equal),
            'less': (np.less, dsc.less),
            'less_equal': (np.less_equal, dsc.less_equal),
            'greater': (np.greater, dsc.greater),
            'greater_equal': (np.greater_equal, dsc.greater_equal),

        }
        for op_name in ops.keys():
            np_op, dsc_op = ops[op_name]
            for dtype in DTYPES:
                if op_name == 'sub':
                    np_op = np.bitwise_xor if is_bool(dtype) else np.subtract

                print(f'Testing operator {op_name} with {dtype.__name__}')
                shape = [randint(2, 10) for _ in range(4)]
                x = random_nd(shape, dtype=dtype)
                x_dsc = dsc.from_numpy(x)

                # Same shape
                y = random_nd(shape, dtype=dtype)
                y_dsc = dsc.from_numpy(y)

                res_np = np_op(x, y)
                res_dsc = dsc_op(x_dsc, y_dsc)
                r_res_np = np_op(y, x)
                r_res_dsc = dsc_op(y_dsc, x_dsc)
                assert all_close(res_dsc, res_np)
                assert all_close(r_res_dsc, r_res_np)

                # Broadcasting
                collapse_idx = randint(0, 3)
                shape[collapse_idx] = 1
                y_b = random_nd(shape, dtype=dtype)
                y_dsc_b = dsc.from_numpy(y_b)
                res_np_b = np_op(x, y_b)
                res_dsc_b = dsc_op(x_dsc, y_dsc_b)
                r_res_np_b = np_op(y_b, x)
                r_res_dsc_b = dsc_op(y_dsc_b, x_dsc)
                assert all_close(res_dsc_b, res_np_b)
                assert all_close(r_res_dsc_b, r_res_np_b)

                # Scalar
                if is_float(dtype):
                    y_s = random()
                elif is_bool(dtype):
                    y_s = bool(randint(0, 1))
                else:
                    y_s = randint(0, 10)

                res_np_s = np_op(x, y_s)
                res_dsc_s = dsc_op(x_dsc, y_s)
                r_res_np_s = np_op(y_s, x)
                r_res_dsc_s = dsc_op(y_s, x_dsc)

                assert all_close(res_dsc_s, res_np_s)
                assert all_close(r_res_dsc_s, r_res_np_s)

    @pytest.mark.skip(reason='not properly implemented yet')
    def test_matmul(self):
        def _mnk() -> tuple[int, int, int]:
            return randint(2, 50), randint(2, 50), randint(2, 50)

        def _test_matmul(shape_a: List[int], shape_b: List[int], dt: np.dtype):
            print(f'Testing {shape_a} @ {shape_b} with {dt.__name__}')
            xa = random_nd(shape_a, dtype=dt)
            xb = random_nd(shape_b, dtype=dt)
            xa_dsc = dsc.from_numpy(xa)
            xb_dsc = dsc.from_numpy(xb)

            res = xa @ xb
            res_dsc = xa_dsc @ xb_dsc
            assert all_close(res_dsc, res)

        for dtype in DSC_DTYPES:
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
            'cos': (np.cos, dsc.cos),
            'tanh': (np.tanh, dsc.tanh),
            'exp': (np.exp, dsc.exp),
            'sqrt': (np.sqrt, dsc.sqrt),
        }
        for op_name in ops.keys():
            np_op, dsc_op = ops[op_name]
            for dtype in DTYPES:
                print(f'Testing {op_name} with {dtype.__name__}')
                x = random_nd([randint(1, 10) for _ in range(4)], dtype=dtype)
                x_dsc = dsc.from_numpy(x)

                res_np = np_op(x)
                res_dsc = dsc_op(x_dsc)
                # There are precision issues when working with non-float types
                assert all_close(res_dsc, res_np, 1e-5 if is_float(dtype) else 1e-3)

    def test_unary_axis(self):
        ops = {
            'sum': (np.sum, dsc.sum),
            'mean': (np.mean, dsc.mean),
            'var': (np.var, dsc.var),
            'max': (np.max, dsc.max),
            'min': (np.min, dsc.min),
        }
        for op_name in ops.keys():
            np_op, dsc_op = ops[op_name]
            for dtype in DTYPES:
                for axis in range(-4, 4):
                    print(f'Testing {op_name} with {dtype.__name__} along axis {axis}')
                    x = random_nd(
                        [randint(1, 10) for _ in range(4)], dtype=dtype
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
            n = randint(1, 10_000)
            for dtype in DTYPES:
                if is_bool(dtype):
                    continue
                print(f'Tensing arange with N={n} and dtype={dtype.__name__} ')
                res_np = np.arange(n, dtype=dtype)
                res_dsc = dsc.arange(n, dtype=DSC_DTYPES[dtype])
                assert all_close(res_dsc, res_np)

    def test_random(self):
        for _ in range(10):
            shape = tuple([randint(1, 10) for _ in range(4)])
            for dtype in DTYPES:
                if not is_float(dtype):
                    continue
                print(f'Tensing randn with dtype={dtype.__name__} ')

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
                        idx = tuple(randint(-10, 9) for _ in range(indexes + 1))
                        res = x[idx]
                        res_dsc = x_dsc[idx]
                        if isinstance(res_dsc, dsc.Tensor):
                            assert all_close(res_dsc, res)
                        else:
                            assert np.isclose(res, res_dsc)
    def test_get_tensor(self):
        for dtype in DTYPES:
            rows = randint(1, 100)
            cols = randint(1, 100)
            x = random_nd([rows, cols], dtype=dtype)
            x_dsc = dsc.from_numpy(x)

            indexes = np.array([randint(0, rows - 1) for _ in range(randint(1, rows))]).astype(np.int32)
            indexes_dsc = dsc.from_numpy(indexes)

            res = x[indexes]
            res_dsc = x_dsc[indexes_dsc]
            assert all_close(res_dsc, res)



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
                        idx = tuple(randint(-10, 9) for _ in range(indexes))
                        val = (
                            random() + 1
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
            shape = tuple(randint(1, 20) for _ in range(n_dim + 1))
            fill = random()

            x = np.full(shape, fill_value=fill, dtype=dtype)
            x_dsc = dsc.full(shape, fill_value=fill, dtype=DSC_DTYPES[dtype])
            assert all_close(x_dsc, x)

            like = np.ones([randint(1, 10) for _ in range(n_dim + 1)])

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
            shape = [randint(2, 10) for _ in range(n_dim)]
            for axis_idx in range(n_dim):
                print(
                    f'Testing concat with {n_dim}-dimensional tensors of type {dtype.__name__} on axis {axis_idx}'
                )
                shape_x1 = list(shape)
                shape_x1[axis_idx] = randint(2, 10)
                shape_x2 = list(shape)
                shape_x2[axis_idx] = randint(2, 10)
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


def test_split():
    for n_dim in range(1, 5):
        for dtype in DTYPES:
            shape = [randint(2, 10) for _ in range(n_dim)]
            for axis_idx in range(n_dim):
                print(f'Testing split with {n_dim}-dimensional tensors of type {dtype.__name__} on axis {axis_idx}')
                ne = shape[axis_idx]
                multi = randint(1, 10)
                shape[axis_idx] *= multi
                x = random_nd(shape, dtype)
                x_dsc = dsc.from_numpy(x)

                res = np.split(x, multi, axis=axis_idx)
                res_dsc = dsc.split(x_dsc, ne, axis=axis_idx)
                assert len(res) == len(res_dsc)
                for r_np, r_dsc in zip(res, res_dsc):
                    assert all_close(r_dsc, r_np)


def test_transpose():
    for n_dim in range(2, 5):
        for dtype in DTYPES:
            print(
                f'Testing transpose with {n_dim}-dimensional tensors of type {dtype.__name__}'
            )
            shape = [randint(2, 10) for _ in range(n_dim)]
            x = random_nd(shape, dtype)
            x_dsc = dsc.from_numpy(x)
            # Simple transpose
            res_np_simple = np.transpose(x)
            res_dsc_simple = dsc.transpose(x_dsc)
            assert all_close(res_dsc_simple, res_np_simple)

            # Test with all the permutations of axes, both positive and negative
            for axes in permutations(range(-n_dim, 0), n_dim):
                res_np = np.transpose(x, axes)
                res_dsc = dsc.transpose(x_dsc, axes)
                assert all_close(res_dsc, res_np)

            for axes in permutations(range(0, n_dim), n_dim):
                res_np = np.transpose(x, axes)
                res_dsc = dsc.transpose(x_dsc, axes)
                assert all_close(res_dsc, res_np)


def test_tril():
    for n_dim in range(2, 5):
        for dtype in DTYPES:
            x = random_nd([randint(1, 10) for _ in range(n_dim)], dtype)
            x_dsc = dsc.from_numpy(x)
            for k in range(-1, 2):
                print(f'Testing tril with {n_dim}-dimensional tensors of type {dtype.__name__} k={k}')
                res = np.tril(x, k)
                res_dsc = dsc.tril(x_dsc, k)
                assert all_close(res_dsc, res)


def test_masked_fill():
    for n_dim in range(1, 5):
        for dtype in DTYPES:
            if not is_float(dtype):
                continue

            print(f'Testing masked_fill with {n_dim}-dimensional tensors of type {dtype.__name__}')
            x = random_nd([randint(1, 10) for _ in range(n_dim)], dtype)
            mask = random_nd(x.shape, np.bool)
            x_dsc = dsc.from_numpy(x)
            mask_dsc = dsc.from_numpy(mask)
            fill = random()

            x[mask] = fill
            res_dsc = x_dsc.masked_fill(mask_dsc, fill)
            assert all_close(res_dsc, x)


def test_topk():
    k = 3
    # Test against torch as it has the same simple API as DSC in this case
    import torch

    def _validate_axis(x: torch.Tensor, x_dsc: dsc.Tensor, axis: int, largest: bool):
        res_v, res_i = torch.topk(x, k, dim=axis, largest=largest)
        res_dsc_v, res_dsc_i = dsc.topk(x_dsc, k, axis=axis, largest=largest)
        assert all_close(res_dsc_v, res_v.detach().cpu().numpy())
        if x.dtype != torch.int32:
            # Indexes returned by topk can be different when working with int tensors, just check the values
            assert all_close(res_dsc_i, res_i.detach().cpu().numpy())

    for n_dim in range(1, 5):
        for dtype in DTYPES:
            if is_bool(dtype):
                continue
            print(f'Testing topk with {n_dim}-dimensional tensors of type {dtype.__name__}')
            x = random_nd([randint(5, 6) for _ in range(n_dim)], dtype)
            x_torch = torch.from_numpy(x)
            x_dsc = dsc.from_numpy(x)

            for axis in range(-n_dim, 0):
                _validate_axis(x_torch, x_dsc, axis, True)
                _validate_axis(x_torch, x_dsc, axis, False)

            for axis in range(0, n_dim):
                _validate_axis(x_torch, x_dsc, axis, True)
                _validate_axis(x_torch, x_dsc, axis, False)
