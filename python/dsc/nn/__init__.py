# Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
# All rights reserved.
#
# This code is licensed under the terms of the 3-clause BSD license
# (https://opensource.org/license/bsd-3-clause).

from ..tensor import Tensor, sum, max, tanh, exp, power
from ..dtype import Dtype
from ..device import Device
from .._bindings import _dsc_new_tensor
from ..context import _get_ctx
from typing import Iterator, Iterable, Any, Tuple, Callable, Optional, OrderedDict, Mapping
from abc import ABC, abstractmethod
import torch
import math
from numpy import ndarray


class Parameter(Tensor):
    def __init__(self, shape: Tuple[int, ...], dtype: Dtype = Dtype.F32, device: Device = Device.DEFAULT, post_init: Optional[Callable[[Tensor], Tensor]] = None):
        super().__init__(_dsc_new_tensor(_get_ctx(), shape, dtype, device))
        self._post_init = post_init
    
    def load(self, x: ndarray):
        super().load(x)
        if self._post_init:
            self = self._post_init(self)


class Module(ABC):
    def __init__(self):
        super().__init__()
        self._parameters = {}
        self._modules = {}

    def register_parameter(self, name: str, param: Parameter):
        if name in self._parameters:
            raise RuntimeError(f'parameter "{name}" already registered')

        self._parameters[name] = param

    def register_module(self, name: str, module: 'Module'):
        if name in self._modules:
            raise RuntimeError(f'module "{name}" already registered')

        self._modules[name] = module

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Parameter):
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            self.register_module(name, value)

        super().__setattr__(name, value)

    def parameters(self) -> Iterator[Parameter]:
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self, prefix: str = '') -> Iterator[Tuple[str, Parameter]]:
        for name, param in self._parameters.items():
            yield prefix + ('.' if prefix else '') + name, param
        for module_name, module in self._modules.items():
            submodule_prefix = prefix + ('.' if prefix else '') + module_name
            yield from module.named_parameters(submodule_prefix)

    def state_dict(self) -> OrderedDict[str, Parameter]:
        res = OrderedDict()
        for name, param in self.named_parameters():
            res[name] = param
        return res

    def from_torch(self, torch_model: torch.nn.Module):
        model_dict = torch_model.state_dict()
        for name, param in self.named_parameters():
            if name not in model_dict:
                print(f'{name} not found in PyTorch model')
                continue
            tensor = model_dict[name]
            param.load(tensor.detach().cpu().numpy())
        
        del model_dict

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        pass

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

class ModuleList(Module):
    def __init__(self, modules: Iterable[Module]):
        super().__init__()
        for i, module in enumerate(modules):
            self.register_module(str(i), module)
    
    def __len__(self) -> int:
        return len(self._modules)
    
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def forward(self):
        raise NotImplementedError('forward() is not supported in ModuleList')

class ModuleDict(Module):
    def __init__(self, modules: Mapping[str, Module]):
        super().__init__()
        for name, module in modules.items():
            setattr(self, name, module)

    def __len__(self) -> int:
        return len(self._modules)
    
    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)
    
    def forward(self):
        raise NotImplementedError('forward() is not supported in ModuleDict')

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = Parameter((out_features, in_features), post_init=Linear._post_init)
        self.bias = Parameter((out_features, )) if bias else None

    @staticmethod
    def _post_init(x: Tensor) -> Tensor:
        return x.transpose()

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.transpose()
        if self.bias:
            out += self.bias
        return out

class LayerNorm(Module):
    def __init__(self, n_features: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.weight = Parameter((n_features, ))
        self.bias = Parameter((n_features, ))
    
    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)

        out = (x - mean) / (var + self.epsilon) ** 0.5
        out = out * self.weight + self.bias
    
        return out

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_size: int):
        super().__init__()
        self.weight = Parameter((num_embeddings, embedding_size))
    
    def forward(self, x: Tensor) -> Tensor:
        return self.weight[x]


def gelu(x: Tensor) -> Tensor:
    return 0.5 * x * (1.0 + tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * power(x, 3))))

def softmax(x: Tensor, axis: int = -1) -> Tensor:
    e = exp((x - max(x, axis=axis, keepdims=True)))
    sum_e = sum(e, axis=axis, keepdims=True)
    return e / sum_e
