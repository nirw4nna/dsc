# Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
# All rights reserved.
#
# This code is licensed under the terms of the 3-clause BSD license
# (https://opensource.org/license/bsd-3-clause).

from dsc.context import init, set_default_device
from dsc.tensor import (
    Tensor,
    from_numpy,
    reshape,
    concat,
    transpose,
    arange,
    randn,
    cos,
    sin,
    sinc,
    logn,
    log2,
    log10,
    exp,
    sqrt,
    absolute,
    angle,
    conj,
    real,
    imag,
    add,
    sub,
    mul,
    true_div,
    sum,
    mean,
    max,
    min,
    clip,
    power,
    i0,
    ones,
    ones_like,
    zeros,
    zeros_like,
    full,
    full_like,
    empty,
    empty_like,
)
from dsc.dtype import Dtype
from dsc.profiler import profile, start_recording, stop_recording
from dsc.device import Device
import dsc.cuda as cuda