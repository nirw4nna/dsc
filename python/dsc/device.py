#  Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
#  All rights reserved.
#
#  This code is licensed under the terms of the 3-clause BSD license
#  (https://opensource.org/license/bsd-3-clause).

from enum import Enum
from typing import Union


DeviceType = Union['Device', str]

class Device(Enum):
    DEFAULT = -1
    CPU = 0
    CUDA = 1

    def __repr__(self) -> str:
        return DEVICE_LOOKUP[self]

    def __str__(self) -> str:
        return repr(self)


def _get_device(dev: DeviceType) -> Device:
    if isinstance(dev, Device):
        return dev
    else:
        if dev in DEVICE_REVERSE_LOOKUP:
            return DEVICE_REVERSE_LOOKUP[dev]
        else:
            raise RuntimeError(f'string "{dev}" is not a valid Device')


DEVICE_VALUE_LOOKUP = {val.value: val for val in Device.__members__.values()}

DEVICE_LOOKUP = {
    Device.DEFAULT: 'default',
    Device.CPU: 'cpu',
    Device.CUDA: 'cuda',
}

DEVICE_REVERSE_LOOKUP = {val: key for key, val in DEVICE_LOOKUP.items()}