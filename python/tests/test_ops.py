import python.dsc as dsc
import pytest
import numpy as np
import time


@pytest.fixture(scope="session", autouse=True)
def session_fixture():
    dsc.init(1024*1024*1024)
    yield


def teardown_function():
    dsc.clear()


def all_close(actual, target, eps=1e-5):
    diffs = ~np.isclose(actual, target, atol=eps, rtol=eps, equal_nan=True)
    close = len(actual[diffs]) == 0
    if not close:
        print(f'Wrong indexes: {np.where(diffs == True)}')
    return close


def test_vec_vec():
    ops_to_test = {

    }


def test_mul():
    a = np.ones(1, np.float64)
    a_dsc = dsc.from_numpy(a, 'A')
    b = a * 0.5
    b_dsc = a_dsc * 0.5
    assert all_close(b, b_dsc.numpy())

