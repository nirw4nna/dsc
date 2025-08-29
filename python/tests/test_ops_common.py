# Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
# All rights reserved.
#
# This code is licensed under the terms of the 3-clause BSD license
# (https://opensource.org/license/bsd-3-clause).

import pytest
from random import randint, random
from itertools import permutations
from utils_cpu import *


@pytest.fixture(scope='session', autouse=True)
def session_fixture():
    # This is invoked once before starting the test session
    dsc.init(int(2**30))
    print(f'Running tests on {DEVICE}')
    dsc.set_default_device(DEVICE)
    yield


@pytest.fixture(autouse=True)
def teardown_fixture():
    # This is invoked automatically after each test
    yield

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
            for axis_idx in range(n_dim):
                shape = [randint(2, 10) for _ in range(n_dim)]
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

def test_repeat():
    for n_dim in range(1, 5):
        for dtype in DTYPES:
            shape = [randint(2, 10) for _ in range(n_dim)]
            for axis_idx in range(n_dim):
                print(f'Testing repeat with {n_dim}-dimensional tensors of type {dtype.__name__} on axis {axis_idx}')
                x = random_nd(shape, dtype)
                x_dsc = dsc.from_numpy(x)
                repeats = randint(2, 5)
                res = np.repeat(x, repeats, axis=axis_idx)
                res_dsc = dsc.repeat(x_dsc, repeats, axis=axis_idx)
                assert all_close(res_dsc, res)

def test_where():
    for n_dim in range(1, 5):
        for dtype in DTYPES:
            print(f'Testing where with {n_dim}-dimensional condition tensor and values of type {dtype.__name__}')
            x = np.random.choice([True, False], size=tuple([randint(1, 10) for _ in range(n_dim)]))
            values = random_nd([2], dtype=dtype)
            this = values[0]; that = values[1]
            x_dsc = dsc.from_numpy(x)
            res = np.where(x, this, that)
            res_dsc = dsc.where(x_dsc, this, that)
            assert all_close(res_dsc, res)

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
