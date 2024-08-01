import python.dsc as dsc
import numpy as np
import random
import pytest


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
            x = np.random.randn(*tuple(shape)).astype(np.float64)
            x_dsc = dsc.from_numpy(x)

            x_np_fft = np.fft.fft(x, n=fft_n, axis=axis)
            x_dsc_fft = dsc.fft(x_dsc, n=fft_n, axis=axis)

            assert all_close(x_dsc_fft.numpy(), x_np_fft)

            x_np_ifft = np.fft.ifft(x_np_fft, axis=axis)
            x_dsc_ifft = dsc.ifft(x_dsc_fft, axis=axis)

            assert all_close(x_dsc_ifft.numpy(), x_np_ifft)

            dsc.clear()
