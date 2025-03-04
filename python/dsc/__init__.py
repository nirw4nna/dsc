# Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
# All rights reserved.
#
# This code is licensed under the terms of the 3-clause BSD license
# (https://opensource.org/license/bsd-3-clause).

from dsc.context import init
from dsc.tensor import (
    Tensor,
    from_numpy,
    reshape,
    concat,
    split,
    transpose,
    tril,
    arange,
    randn,
    cos,
    sin,
    tanh,
    exp,
    sqrt,
    add,
    sub,
    mul,
    true_div,
    sum,
    mean,
    var,
    matmul,
    max,
    min,
    power,
    equal,
    not_equal, 
    less,
    less_equal,
    greater,
    greater_equal,
    ones,
    ones_like,
    zeros,
    zeros_like,
    full,
    full_like,
    empty,
    empty_like,
    topk,
    multinomial,
)
from dsc.dtype import Dtype
from dsc.profiler import profile, start_recording, stop_recording
from dsc.device import Device
import dsc.nn as nn