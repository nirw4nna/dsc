# Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
# All rights reserved.
#
# This code is licensed under the terms of the 3-clause BSD license
# (https://opensource.org/license/bsd-3-clause).

import pytest
from random import randint, random
from utils_cpu import *


@pytest.fixture(scope='session', autouse=True)
def session_fixture():
    # This is invoked once before starting the test session
    dsc.init(int(2**30))
    dsc.set_default_device('cpu')
    yield


@pytest.fixture(autouse=True)
def teardown_fixture():
    # This is invoked automatically after each test
    yield


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

    def test_outer(self):
        for dtype in DTYPES:
            for _ in range(10):
                xa = random_nd([randint(2, 50)], dtype)
                xb = random_nd([randint(2, 50)], dtype)
                xa_dsc = dsc.from_numpy(xa)
                xb_dsc = dsc.from_numpy(xb)

                out = np.outer(xa, xb)
                out_dsc = dsc.outer(xa_dsc, xb_dsc)
                assert all_close(out_dsc, out)

    def test_matmul(self):
        def _mnk() -> tuple[int, int, int]:
            return randint(50, 100), randint(50, 100), randint(50, 100)

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
            if is_bool(dtype):
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
