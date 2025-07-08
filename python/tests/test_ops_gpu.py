# Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
# All rights reserved.
#
# This code is licensed under the terms of the 3-clause BSD license
# (https://opensource.org/license/bsd-3-clause).


import torch
import dsc
from random import randint, random
from typing import List
import pytest


DTYPES = [torch.bool, torch.int32, torch.float32, torch.float64]
TORCH_TO_DSC_DTYPES = {
    torch.bool: dsc.bool_,
    torch.int32: dsc.i32,
    torch.float32: dsc.f32,
    torch.float64: dsc.f64,
}

if dsc.gpu.has_bf16():
    DTYPES = [torch.bool, torch.int32, torch.bfloat16, torch.float32, torch.float64]
    TORCH_TO_DSC_DTYPES = {
        torch.bool: dsc.bool_,
        torch.int32: dsc.i32,
        torch.bfloat16: dsc.bf16,
        torch.float32: dsc.f32,
        torch.float64: dsc.f64,
    }

DSC_TO_TORCH_DTYPES = {v: k for k, v in TORCH_TO_DSC_DTYPES.items()}


def is_float(dtype: torch.dtype) -> bool:
    return dtype == torch.bfloat16 or dtype == torch.float32 or dtype == torch.float64


def is_bool(dtype: torch.dtype) -> bool:
    return dtype == torch.bool


def is_integer(dtype: torch.dtype) -> bool:
    return dtype == torch.int32


@pytest.fixture(scope='session', autouse=True)
def session_fixture():
    if not dsc.gpu.is_available():
        pytest.skip('GPU not available - skipping all GPU tests', allow_module_level=True)
    dsc.init(int(2**30))
    # This is invoked once before starting the test session
    dsc.set_default_device('gpu')
    yield


@pytest.fixture(autouse=True)
def teardown_fixture():
    # This is invoked automatically after each test
    yield


def random_nd(shape: List[int], dtype: torch.dtype = torch.float64) -> (torch.Tensor, dsc.Tensor):
    torch_tensor = None
    if dtype == torch.bool:
        torch_tensor = torch.randint(0, 2, size=tuple(shape), device='cuda').to(dtype)
    elif dtype == torch.int32:
        # Return a positive integer tensor if the dtype is int32 so that we don't have issues
        # with power
        torch_tensor = torch.randint(0, 10, size=tuple(shape), device='cuda').to(dtype)
    else:
        torch_tensor = torch.randn(*tuple(shape), device='cuda').to(dtype)

    return (torch_tensor,
            dsc.frombuffer(torch_tensor.shape, TORCH_TO_DSC_DTYPES[torch_tensor.dtype],
                           torch_tensor.data_ptr(), device='gpu', data_device='gpu'))


def all_close(actual: dsc.Tensor, target: torch.Tensor, atol: float = 1e-4, rtol: float = 1e-4) -> bool:
    torch.cuda.synchronize()
    dsc.gpu.synchronize()
    actual_dtype = DSC_TO_TORCH_DTYPES[actual.dtype]
    actual_torch = torch.as_tensor(actual, device='cuda').view(actual_dtype)
    if is_float(actual_dtype) and not is_float(target.dtype):
        target = target.to(dtype=actual_dtype)
    return torch.allclose(actual_torch, target, atol=atol, rtol=rtol, equal_nan=True)


class TestOps:
    def test_binary(self):
        ops = {
            'add': (torch.add, dsc.add),
            'sub': (torch.subtract, dsc.sub),
            'mul': (torch.multiply, dsc.mul),
            'div': (torch.true_divide, dsc.true_div),
            'pow': (torch.pow, dsc.power),
            'equal': (torch.eq, dsc.equal),
            'not_equal': (torch.ne, dsc.not_equal),
            'less': (torch.lt, dsc.less),
            'less_equal': (torch.le, dsc.less_equal),
            'greater': (torch.gt, dsc.greater),
            'greater_equal': (torch.ge, dsc.greater_equal),
        }
        for op_name in ops.keys():
            torch_op, dsc_op = ops[op_name]
            for dtype in DTYPES:
                if op_name == 'sub':
                    torch_op = torch.bitwise_xor if is_bool(dtype) else torch.subtract
                if op_name == 'pow' and is_bool(dtype):
                    # Pow on CUDA is not implemented in torch for bool
                    continue

                atol = 1e-4; rtol = 1e-4
                if dtype == torch.bfloat16:
                    atol = 1e-1; rtol = 1e-2
                print(f'Testing operator {op_name} with {dtype}')
                shape = [randint(2, 10) for _ in range(4)]
                x, x_dsc = random_nd(shape, dtype=dtype)

                # Same shape
                y, y_dsc = random_nd(shape, dtype=dtype)

                res_torch = torch_op(x, y)
                res_dsc = dsc_op(x_dsc, y_dsc)
                r_res_torch = torch_op(y, x)
                r_res_dsc = dsc_op(y_dsc, x_dsc)
                assert all_close(res_dsc, res_torch, atol, rtol), f'Error testing ({x.shape} {op_name} {y.shape}) dtype={dtype}'
                assert all_close(r_res_dsc, r_res_torch, atol, rtol), f'Error testing ({y.shape} {op_name} {x.shape}) dtype={dtype}'

                # Broadcasting
                collapse_idx = randint(0, 3)
                shape[collapse_idx] = 1
                y_b, y_dsc_b = random_nd(shape, dtype=dtype)

                res_torch_b = torch_op(x, y_b)
                res_dsc_b = dsc_op(x_dsc, y_dsc_b)
                r_res_torch_b = torch_op(y_b, x)
                r_res_dsc_b = dsc_op(y_dsc_b, x_dsc)
                assert all_close(res_dsc_b, res_torch_b, atol, rtol), f'Error testing ({x.shape} {op_name} {y_b.shape}) dtype={dtype}'
                assert all_close(r_res_dsc_b, r_res_torch_b, atol, rtol), f'Error testing ({y_b.shape} {op_name} {x.shape}) dtype={dtype}'

                # Scalar
                if is_float(dtype):
                    y_s = random()

                elif is_bool(dtype):
                    y_s = bool(randint(0, 1))
                else:
                    y_s = randint(0, 10)

                res_torch_s = torch_op(x, y_s)
                res_dsc_s = dsc_op(x_dsc, y_s)
                if 'equal' in op_name or op_name == 'less' or op_name == 'greater':
                    # For comparison ops torch requires the first argument to be a tensor
                    continue
                r_res_torch_s = torch_op(y_s, x)
                r_res_dsc_s = dsc_op(y_s, x_dsc)
                assert all_close(res_dsc_s, res_torch_s, atol, rtol), f'Error testing ({x.shape} {op_name} {y_s}) dtype={dtype}'
                assert all_close(r_res_dsc_s, r_res_torch_s, atol, rtol), f'Error testing ({y_s} {op_name} {x.shape}) dtype={dtype}'

    def test_unary(self):
        ops = {
            'sin': (torch.sin, dsc.sin),
            'cos': (torch.cos, dsc.cos),
            'tanh': (torch.tanh, dsc.tanh),
            'exp': (torch.exp, dsc.exp),
            'sqrt': (torch.sqrt, dsc.sqrt),
        }
        for op_name in ops.keys():
            torch_op, dsc_op = ops[op_name]
            for dtype in DTYPES:
                print(f'Testing {op_name} with {dtype}')
                x, x_dsc = random_nd([randint(1, 10) for _ in range(4)], dtype=dtype)

                res_torch = torch_op(x)
                res_dsc = dsc_op(x_dsc)

                assert all_close(res_dsc, res_torch), f'Error testing {op_name} shape={x.shape} dtype={dtype}'
    
    def test_unary_axis(self):
        ops = {
            'sum': (torch.sum, dsc.sum),
            'mean': (torch.mean, dsc.mean),
            'var': (torch.var, dsc.var),
            'max': (torch.amax, dsc.max),
            'min': (torch.amin, dsc.min),
        }
        for op_name in ops.keys():
            torch_op, dsc_op = ops[op_name]
            for dtype in DTYPES:
                for axis in range(-4, 4):
                    rtol = 1e-4; atol = 1e-4
                    params_torch = {
                        'dim': axis,
                        'keepdim': True
                    }
                    if op_name == 'mean' or op_name == 'var':
                        if not is_float(dtype) or dtype == torch.bfloat16:
                            continue
                        atol = 1e-3; rtol = 1e-2
                        if op_name == 'var':
                            params_torch['correction'] = 0

                    print(f'Testing {op_name} with {dtype} along axis {axis}')
                    x, x_dsc = random_nd(
                        [randint(1, 10) for _ in range(4)], dtype=dtype
                    )

                    res_torch = torch_op(x, **params_torch)
                    res_dsc = dsc_op(x_dsc, axis=axis, keepdims=True)
                    assert all_close(res_dsc, res_torch, atol, rtol), f'Error testing {op_name} shape={x.shape} dtype={x.dtype} keepdims=True'

                    params_torch['keepdim'] = False
                    res_torch_2 = torch_op(x, **params_torch)
                    res_dsc_2 = dsc_op(x_dsc, axis=axis, keepdims=False)
                    assert all_close(res_dsc_2, res_torch_2, atol, rtol), f'Error testing {op_name} shape={x.shape} dtype={x.dtype} keepdims=False'

    def test_matmul(self):
        def _mnk() -> tuple[int, int, int]:
            return randint(50, 100), randint(50, 100), randint(50, 100)

        def _test_matmul(shape_a: List[int], shape_b: List[int], dt: torch.dtype):
            print(f'Testing {shape_a} @ {shape_b} on with {dt}')
            xa, xa_dsc = random_nd(shape_a, dtype=dt)
            xb, xb_dsc = random_nd(shape_b, dtype=dt)
            res = xa @ xb
            res_dsc = xa_dsc @ xb_dsc
            # TODO: it looks like BF16 has a lower precision, should check what torch actually does. For now fix the tolerance at 1%
            assert all_close(res_dsc, res, atol=1e-1, rtol=1e-2), f'Error testing {shape_a} @ {shape_b} with {dt}'

        for dtype in DTYPES:
            if not is_float(dtype):
                continue
            # 2D GEMM
            m, n, k = _mnk()
            _test_matmul([m, k], [k, n], dtype)
            # GEVM
            _test_matmul([1, k], [k, n], dtype)

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
    def test_outer(self):
        for dtype in DTYPES:
            for _ in range(10):
                xa, xa_dsc = random_nd([randint(2, 50)], dtype)
                xb, xb_dsc = random_nd([randint(2, 50)], dtype)

                out = torch.outer(xa, xb)
                out_dsc = dsc.outer(xa_dsc, xb_dsc)
                assert all_close(out_dsc, out)
