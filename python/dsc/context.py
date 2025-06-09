# Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
# All rights reserved.
#
# This code is licensed under the terms of the 3-clause BSD license
# (https://opensource.org/license/bsd-3-clause).

from ._bindings import _dsc_ctx_init, _dsc_print_mem_usage, _dsc_set_default_device
from .device import _get_device, DeviceType
import psutil

_ctx_instance = None


class _DscContext:
    def __init__(self, main_mem: int):
        self._ctx = _dsc_ctx_init(main_mem)

    # TODO: (3)
    # def __del__(self):
        # _dsc_ctx_free(self._ctx)


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

def print_mem_usage():
    _dsc_print_mem_usage(_get_ctx())

def set_default_device(device: DeviceType):
    _dsc_set_default_device(_get_ctx(), _get_device(device))