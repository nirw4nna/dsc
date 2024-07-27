import python.dsc as dsc
import numpy as np
import time


def bench_ops():
    pass

# def test_vec_vec():
#     M = 64
#     N = 2 ** 16
#     a = np.random.rand(M, N).astype(np.complex128)
#     b = np.random.rand(N).astype(np.complex128)
#     a_dsp = dsc.from_numpy(a, 'A')
#     b_dsp = dsc.from_numpy(b, 'B')
#
#     delay_np = float('+inf')
#     delay_dsc = float('+inf')
#     for _ in range(5):
#         start = time.perf_counter()
#         c = a * b
#         this_delay = time.perf_counter() - start
#         delay_np = this_delay if this_delay < delay_np else delay_np
#
#     for _ in range(5):
#         start = time.perf_counter()
#         c_dsp = a_dsp * b_dsp
#         this_delay = time.perf_counter() - start
#         delay_dsc = this_delay if this_delay < delay_dsc else delay_dsc
#
#     assert all_close(c, c_dsp.numpy())
#
#     print(f'NumPy took {round(delay_np * 1e3, 2)}ms DSPCraft took {round(delay_dsc* 1e3, 2)}ms')
#
