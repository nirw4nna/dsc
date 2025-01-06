#  Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
#  All rights reserved.
#
#  This code is licensed under the terms of the 3-clause BSD license
#  (https://opensource.org/license/bsd-3-clause).

from .._bindings import (
    _dsc_cuda_available,
    _dsc_cuda_devices,
    _dsc_cuda_set_device,
    _dsc_cuda_dev_capability,
    _dsc_cuda_dev_mem,
    _dsc_cuda_sync,
)
from ..context import _get_ctx


def is_available() -> bool:
    return _dsc_cuda_available(_get_ctx())

def device_count() -> int:
    return _dsc_cuda_devices(_get_ctx())

def set_device(device: int):
    _dsc_cuda_set_device(_get_ctx(), device)

def get_device_capability(device: int) -> int:
    return _dsc_cuda_dev_capability(_get_ctx(), device)

def get_device_mem(device: int) -> int:
    return _dsc_cuda_dev_mem(_get_ctx(), device)

def synchronize():
    _dsc_cuda_sync(_get_ctx())