# Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
# All rights reserved.
#
# This code is licensed under the terms of the 3-clause BSD license
# (https://opensource.org/license/bsd-3-clause).

import pytest
from utils_cpu import *
from random import randint, random
from typing import List
import math


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
            # Indexes are always on CPU
            indexes_dsc = dsc.from_numpy(indexes, device='cpu')

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