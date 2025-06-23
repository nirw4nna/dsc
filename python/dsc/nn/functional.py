#  Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
#  All rights reserved.
#
#  This code is licensed under the terms of the 3-clause BSD license
#  (https://opensource.org/license/bsd-3-clause).


from ..tensor import Tensor, tanh, power, max, sum, exp
from ..profiler import trace
import math


@trace('gelu')
def gelu(x: Tensor) -> Tensor:
    return 0.5 * x * (1.0 + tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * power(x, 3))))


@trace('softmax')
def softmax(x: Tensor, axis: int = -1) -> Tensor:
    e = exp((x - max(x, axis=axis, keepdims=True)))
    sum_e = sum(e, axis=axis, keepdims=True)
    return e / sum_e


@trace('silu')
def silu(x: Tensor) -> Tensor:
    return x * (1 / (1 + exp(-x)))