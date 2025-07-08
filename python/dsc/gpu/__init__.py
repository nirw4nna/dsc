#  Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
#  All rights reserved.
#
#  This code is licensed under the terms of the 3-clause BSD license
#  (https://opensource.org/license/bsd-3-clause).

from .._bindings import (
    _dsc_get_gpu_platform,
    _dsc_gpu_available,
    _dsc_gpu_devices,
    _dsc_gpu_set_device,
    _dsc_gpu_dev_capability,
    _dsc_gpu_dev_mem,
    _dsc_gpu_sync,
    _dsc_gpu_has_bf16,
    _DSC_PLATFORM_CUDA,
    _DSC_PLATFORM_ROCM,
)


def is_available() -> bool:
    return _dsc_gpu_available(None)

def is_cuda() -> bool:
    return _dsc_get_gpu_platform(None) == _DSC_PLATFORM_CUDA

def is_rocm() -> bool:
    return _dsc_get_gpu_platform(None) == _DSC_PLATFORM_ROCM

def has_bf16() -> bool:
    return _dsc_gpu_has_bf16(None)

def device_count() -> int:
    return _dsc_gpu_devices(None)

def set_device(device: int):
    _dsc_gpu_set_device(None, device)

def get_device_capability(device: int) -> int:
    return _dsc_gpu_dev_capability(None, device)

def get_device_mem(device: int) -> int:
    return _dsc_gpu_dev_mem(None, device)

def synchronize():
    _dsc_gpu_sync(None)