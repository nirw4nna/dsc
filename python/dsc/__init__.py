# Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
# All rights reserved.
#
# This code is licensed under the terms of the 3-clause BSD license
# (https://opensource.org/license/bsd-3-clause).

from dsc.context import init, print_mem_usage, set_default_device
from dsc.tensor import (
    Tensor,
    from_numpy,
    from_buffer,
    reshape,
    concat,
    split,
    transpose,
    tril,
    arange,
    repeat,
    randn,
    cos,
    sin,
    tanh,
    exp,
    sqrt,
    rsqrt,
    add,
    sub,
    mul,
    true_div,
    sum,
    mean,
    var,
    matmul,
    outer,
    max,
    min,
    power,
    equal,
    not_equal, 
    less,
    less_equal,
    greater,
    greater_equal,
    tensor,
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
    where,
)
from dsc.dtype import Dtype, bool_, i32, f32, f64
from dsc.profiler import start_recording, stop_recording, profile, trace
from dsc.device import Device
import dsc.cuda as cuda
import dsc.nn as nn