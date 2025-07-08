# Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
# All rights reserved.
#
# This code is licensed under the terms of the 3-clause BSD license
# (https://opensource.org/license/bsd-3-clause).

from ..tensor import Tensor, power, matmul, rsqrt
from ..dtype import Dtype
from ..device import Device
from .._bindings import _dsc_new_tensor
from ..context import _get_ctx
from ..profiler import trace
from typing import Iterator, Dict, Iterable, Any, Tuple, Callable, Optional, OrderedDict, Mapping, List
from abc import ABC, abstractmethod
from tqdm import tqdm
from .utils import safe_load
from . import functional


class Parameter(Tensor):
    def __init__(self, shape: Tuple[int, ...], dtype: Dtype = Dtype.F32, device: Device = Device.DEFAULT, on_load: Optional[Callable[[Tensor], Tensor]] = None):
        # Parameters are lazy tensors (i.e. Tensors that don't have an underlying buffer)
        super().__init__(_dsc_new_tensor(_get_ctx(), shape, dtype, device, True))
        self._on_load = on_load

    def load(self, x: Tensor):
        if self._on_load is not None:
            x = self._on_load(x)
        super().load(x)

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
        res = OrderedDict[str, Parameter]()
        for name, param in self.named_parameters():
            res[name] = param
        return res

    def from_state(self, state_dict: Dict[str, Tensor],
                   on_hook: Optional[List[Tuple[List[str], Callable[[Tensor], Tensor]]]] = None,
                   tied: Optional[Dict[str, str]] = None):
        with tqdm(total=len(state_dict), desc='Loading model parameters') as pbar:
            for name, param in self.named_parameters():
                real_name = name
                if tied is not None and name in tied:
                    name = tied[name]

                if name not in state_dict:
                    pbar.write(f'{name} not found in DSC model')
                    pbar.update(1)
                    continue

                tensor = state_dict[name]
                if on_hook:
                    # on_hook defines transformations on tensors that are called before loading the tensors in DSC
                    for keys, hook in on_hook:
                        # If any of the keys starts with 'name' I'll apply the hook
                        if any(name.endswith(key) for key in keys):
                            tensor = hook(tensor)

                pbar.set_description(f'{real_name if real_name == name else f"{real_name} (tied to {name})"} {tensor.shape} {tensor.dtype}')
                param.load(tensor)
                pbar.update(1)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
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
    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype: Dtype = Dtype.F32):
        super().__init__()
        self.weight = Parameter((out_features, in_features), dtype=dtype)
        self.bias = Parameter((out_features, ), dtype=dtype) if bias else None

    @trace('Linear')
    def forward(self, x: Tensor) -> Tensor:
        out = matmul(x, self.weight, trans_b=True)
        if self.bias:
            out += self.bias
        return out

class LayerNorm(Module):
    def __init__(self, n_features: int, epsilon: float = 1e-5, dtype: Dtype = Dtype.F32):
        super().__init__()
        self.epsilon = epsilon
        self.weight = Parameter((n_features, ), dtype=dtype)
        self.bias = Parameter((n_features, ), dtype=dtype)

    @trace('LayerNorm')
    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)

        out = (x - mean) / (var + self.epsilon) ** 0.5
        out = out * self.weight + self.bias
    
        return out

class RMSNorm(Module):
    def __init__(self, in_shape: int, epsilon: float = 1e-6, dtype: Dtype = Dtype.F32):
        super().__init__()
        self.epsilon = epsilon
        self.weight = Parameter((in_shape, ), dtype=dtype)

    @trace('RMSNorm')
    def forward(self, x: Tensor) -> Tensor:
        var = power(x, 2).mean(-1, keepdims=True)
        out = x * rsqrt(var + self.epsilon)
        return out * self.weight

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_size: int, dtype: Dtype = Dtype.F32):
        super().__init__()
        self.weight = Parameter((num_embeddings, embedding_size), dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.weight[x]
