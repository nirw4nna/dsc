# Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
# All rights reserved.
#
# This code is licensed under the terms of the 3-clause BSD license
# (https://opensource.org/license/bsd-3-clause).

from ._bindings import _dsc_ctx_init, _dsc_ctx_clear, _dsc_ctx_free
import psutil

_ctx_instance = None


def _get_ctx():
    global _ctx_instance
    if _ctx_instance is None:
        # Workaround: instead of throwing an error if the context is not initialized
        # we can simply initialize one with a fixed amount of memory that is a small %
        # of the total available memory.
        total_mem = psutil.virtual_memory().total
        nb = int(min(total_mem * 0.1, 2**30))
        print(
            f'DSC has not been explicitly initialized! Will create a context of {round(nb / (2**20))} MB. '
            f'If you require more memory please call dsc.init() once before executing your code.'
        )
        _ctx_instance = _DscContext(nb)
    return _ctx_instance._ctx


def init(nb: int):
    global _ctx_instance
    if _ctx_instance is None:
        _ctx_instance = _DscContext(nb)
    else:
        raise RuntimeWarning('Context already initialized')


def clear():
    global _ctx_instance
    if _ctx_instance is not None:
        _ctx_instance.clear()


class _DscContext:
    def __init__(self, nb: int):
        self._ctx = _dsc_ctx_init(nb)

    def __del__(self):
        _dsc_ctx_free(self._ctx)

    def clear(self):
        _dsc_ctx_clear(self._ctx)
