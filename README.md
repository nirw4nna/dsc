<div align="center">
<img src="docs/logo.png" alt="Logo" width="200">

<h1>
DSC
</h1>

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Unit Tests](https://github.com/nirw4nna/dsc/actions/workflows/tests.yml/badge.svg)](https://github.com/nirw4nna/dsc/actions/workflows/tests.yml)

</div>

---

## About
DSC is a PyTorch-compatible tensor library and inference framework for machine learning models.
It features a C-compatible low-level API that is wrapped in a modern Python API very similar to NumPy / PyTorch but
with some nice usability improvements.


Some key features of DSC include:
- **Intuitive API**: DSC Python API closely resembles NumPy / PyTorch.


- **Built-in neural networks support**: DSC comes with `nn.Module` built-in. Porting a model from PyTorch to DSC
is trivial (check out the [examples](https://github.com/nirw4nna/dsc/tree/main/examples/models)).


- **Multiple backends**: DSC supports both **CPU** and **CUDA** with other backends being worked on.
Programs written using DSC can seamlessly switch between backends by simply adding a `dsc.set_default_device('...')`
instruction, no changes needed.


- **Minimal external dependencies**: DSC doesn't require external libraries to be efficient.
On CPU the core operations are written from scratch in portable C++, this makes code written using DSC extremely portable.


- **No runtime allocations**: DSC has it's own custom memory allocator, memory is pre-allocated
only once so no extra calls to `malloc()` or `free()` are required. It's also possible
to switch to a linear allocator to remove the (minimal) overhead introduced by a general purpose allocator.


---


## Quick start
Getting started with DSC is very simple. The only requirements are:
- A compiler with good support for C++20
- GNU Make for building

On a Linux-based system these can be obtained with:
```shell
sudo apt update
sudo apt install build-essential
```

### Installation
The recommended way to install DSC is from source:
```shell
git clone git@github.com:nirw4nna/dsc.git
cd dsc/
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -e .
```

To build the C++ library:
```shell
make clean; make shared DSC_FAST=1
```
This will compile DSC without any debug information, you can specify different options
to enable/disable specific features:

| Option        | Description                                                                  |
|---------------|------------------------------------------------------------------------------|
| DSC_LOG_LEVEL | Configure the logging level (values: [0-3] with 0 meaning everything on)     |
| DSC_FAST      | Turn off logging (level=2) and compile with the highest optimisation level   |
| DSC_CUDA      | Enable CUDA backend                                                          |
| DSC_MAX_OBJS  | Max number of DSC tensors that can be used at the same time (**default=1K**) |

To verify that everything worked out as expected try a simple operation:
```shell
python3 -c "import dsc; x = dsc.arange(10); print(x)"
```

### CUDA backend
This provides GPU acceleration on NVIDIA GPUs. To get started make sure to have the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
installed.

To build the C++ library with CUDA enabled:
```shell
make clean; make shared DSC_FAST=1 DSC_CUDA=1
```

**Note:** if you see errors when compiling with CUDA support make sure that the CUDA installation path is `/usr/local/cuda`
(default on Ubuntu).
If that is not the case you have to manually update the Makefile or set the `CUDA` environment variable before calling make.

To verify that the CUDA backend is working try a simple GPU operation:
```shell
python3 -c "import dsc; x = dsc.arange(10, device='cuda'); print(x)"
```


## Running tests
DSC uses `pytest` to run unit tests against NumPy which is the reference for correctness.

To run all the tests simple do:
```bash
cd python/tests/
pytest -s test_ops.py --no-header --no-summary -q
```
**Note:** there are quite a few tests so to run them it's better to compile DSC with `DSC_FAST=1`.

## License
BSD-3-Clause
