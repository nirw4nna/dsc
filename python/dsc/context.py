# Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
# All rights reserved.
#
# This code is licensed under the terms of the 3-clause BSD license
# (https://opensource.org/license/bsd-3-clause).
from ctypes import c_uint8

from ._bindings import _dsc_ctx_init, _dsc_ctx_free, _dsc_set_default_device
import psutil

_ctx_instance = None


def _get_ctx():
    global _ctx_instance
    if _ctx_instance is None:
        # Workaround: instead of throwing an error if the context is not initialized
        # we can simply initialize one with a fixed amount of memory that is a small %
        # of the total available memory.
        total_mem = psutil.virtual_memory().total
        mem = int(total_mem * 0.1)
        print(
            f'DSC has not been explicitly initialized. Using {round(mem / (1024. * 1024.))}MB.'
            f' If you require more memory please call dsc.init() once before executing your code.'
        )
        _ctx_instance = _DscContext(mem)
    return _ctx_instance._ctx


def init(mem_size: int):
    global _ctx_instance
    if _ctx_instance is None:
        _ctx_instance = _DscContext(mem_size)
    else:
        raise RuntimeWarning('Context already initialized')


def set_default_device(device: int):
    _dsc_set_default_device(_get_ctx(), device)


class _DscContext:
    def __init__(self, main_mem: int):
        self._ctx = _dsc_ctx_init(main_mem)

    def __del__(self):
        _dsc_ctx_free(self._ctx)
