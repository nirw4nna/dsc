import python.dsc as dsc
import numpy as np
import random
import pytest
from typing import List


@pytest.fixture(scope="session", autouse=True)
def session_fixture():
    dsc.init(1024*1024*1024)
    yield


def teardown_function():
    dsc.clear()


def all_close(actual, target, eps=1e-5):
    diffs = ~np.isclose(actual, target, atol=eps, rtol=eps, equal_nan=True)
    close = len(actual[diffs]) == 0
    return close


def random_nd(shape: List[int], dtype: np.dtype = np.float64):
    return np.random.randn(*tuple(shape)).astype(dtype)


DTYPES = [np.float32, np.float64, np.complex64, np.complex128]
DSC_DTYPES = {
    np.float32: dsc.Dtype.F32,
    np.float64: dsc.Dtype.F64,
    np.complex64: dsc.Dtype.C32,
    np.complex128: dsc.Dtype.C64
}


def test_ops():
    ops = {
        'add': (np.add, '+'),
        'sub': (np.subtract, '-'),
        'mul': (np.multiply, '*'),
        'div': (np.true_divide, '/'),
    }
    for op_name in ops.keys():
        np_op, symbol = ops[op_name]
        for dtype in DTYPES:
            print(f'Testing operator {op_name} ({symbol}) with {dtype.__name__}')
            shape = [random.randint(2, 10) for _ in range(4)]

            x = random_nd(shape, dtype=dtype)
            x_dsc = dsc.from_numpy(x)

            # Same shape
            y = random_nd(shape, dtype=dtype)
            y_dsc = dsc.from_numpy(y)

            res_np = np_op(x, y)
            res_dsc = eval(f'x_dsc {symbol} y_dsc')
            assert all_close(res_dsc.numpy(), res_np)

            # Broadcasting
            collapse_idx = random.randint(0, 3)
            shape[collapse_idx] = 1

            y_b = random_nd(shape, dtype=dtype)
            y_dsc_b = dsc.from_numpy(y_b)

            res_np_b = np_op(x, y_b)
            res_dsc_b = eval(f'x_dsc {symbol} y_dsc_b')
            assert all_close(res_dsc_b.numpy(), res_np_b)

            # Scalar
            if dtype == np.complex64 or dtype == np.complex128:
                y_s = complex(random.random(), random.random())
            else:
                y_s = random.random()

            res_np_s = np_op(x, y_s)
            res_dsc_s = eval(f'x_dsc {symbol} y_s')
            assert all_close(res_dsc_s.numpy(), res_np_s)

            dsc.clear()


def test_trig():
    ops = {
        'sin': (np.sin, dsc.sin),
        'cos': (np.cos, dsc.cos),
    }
    for op_name in ops.keys():
        np_op, dsc_op = ops[op_name]
        for dtype in DTYPES:
            print(f'Testing {op_name} with {dtype.__name__}')
            x = random_nd([random.randint(1, 10) for _ in range(4)], dtype=dtype)
            x_dsc = dsc.from_numpy(x)

            res_np = np_op(x)
            res_dsc = dsc_op(x_dsc)
            assert all_close(res_dsc.numpy(), res_np)


def test_fft():
    n_ = random.randint(3, 10)
    n = 2 ** n_

    for axis in range(4):
        shape = [8] * 4
        shape[axis] = n
        for n_change in range(-1, 2):
            # n_change=-1 -> cropping
            # n_change=0  -> copy
            # n_change=+1 -> padding
            fft_n = 2 ** (n_ + n_change)
            x = random_nd(shape)
            x_dsc = dsc.from_numpy(x)

            x_np_fft = np.fft.fft(x, n=fft_n, axis=axis)
            x_dsc_fft = dsc.fft(x_dsc, n=fft_n, axis=axis)

            assert all_close(x_dsc_fft.numpy(), x_np_fft)

            x_np_ifft = np.fft.ifft(x_np_fft, axis=axis)
            x_dsc_ifft = dsc.ifft(x_dsc_fft, axis=axis)

            assert all_close(x_dsc_ifft.numpy(), x_np_ifft)

            dsc.clear()


def test_rfftfreq():
    for _ in range(10):
        n = random.randint(1, 10_000)
        for dtype in DTYPES:
            if dtype == np.complex64 or dtype == np.complex128:
                continue
            print(f'Tensing rfftfreq with N={n} and dtype={dtype.__name__}')

            # With default d
            res_np = np.fft.rfftfreq(n).astype(dtype)
            res_dsc = dsc.rfftfreq(n, dtype=DSC_DTYPES[dtype])
            assert all_close(res_np, res_dsc.numpy())

            # With random d
            d = random.random()
            res_np_d = np.fft.rfftfreq(n, d).astype(dtype)
            res_dsc_d = dsc.rfftfreq(n, d, dtype=DSC_DTYPES[dtype])
            assert all_close(res_np_d, res_dsc_d.numpy())
            dsc.clear()


def test_arange():
    for _ in range(10):
        n = random.randint(1, 10_000)

        for dtype in DTYPES:
            print(f'Tensing arange with N={n} and dtype={dtype.__name__}')
            res_np = np.arange(n, dtype=dtype)
            res_dsc = dsc.arange(n, dtype=DSC_DTYPES[dtype])
            assert all_close(res_dsc.numpy(), res_np)

            dsc.clear()


def test_random():
    for _ in range(10):
        shape = tuple([random.randint(1, 10) for _ in range(4)])
        for dtype in DTYPES:
            if dtype == np.complex64 or dtype == np.complex128:
                continue
            print(f'Tensing randn with dtype={dtype.__name__}')

            res_np = np.random.randn(*shape).astype(dtype)
            res_dsc = dsc.randn(*shape, dtype=DSC_DTYPES[dtype])
            res_dsc_np = res_dsc.numpy()

            assert res_dsc_np.dtype == res_np.dtype
            assert res_dsc_np.shape == res_np.shape

            dsc.clear()
