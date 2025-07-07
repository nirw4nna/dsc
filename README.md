<div align="center">
<img src="docs/logo.png" alt="Logo" width="200">

<h3>
DSC
</h3>

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
| DSC_GPU       | Enable GPU support                                                           |
| DSC_MAX_OBJS  | Max number of DSC tensors that can be used at the same time (**default=1K**) |

To verify that everything worked out as expected try a simple operation:
```shell
python3 -c "import dsc; x = dsc.arange(10); print(x)"
```
When running on the CPU it may be beneficial to use multiple threads for certain operations (i.e. when doing matrix-vector
products). DSC has native support for concurrency, to enable it set the `DSC_NUM_THREADS` environment variable.
If you set it to -1 it will use half of your available cores.

### Notes on GPU support
DSC supports both AMD and NVIDIA GPUs. If compiled with `DSC_GPU=1` it will automatically detect the appropriate backend.
You can see which backend has been selected by checking the output of the Makefile or, once the compilation is done,
use the Python API:
```python
import dsc

if dsc.gpu.is_available(): # If a GPU backend has been detected you can check if it's ROCm or CUDA
    dsc.gpu.is_rocm()
    dsc.gpu.is_cuda()
```

### CUDA backend
This provides GPU acceleration on NVIDIA GPUs. To get started make sure to have the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
installed.

To build the C++ library with CUDA enabled simply specify `DSC_GPU=1`. CUDA will be detected automatically if you installed it.

**Note:** if you see errors when compiling with CUDA support make sure that the CUDA installation path specified in the Makefile
is correct. If this is not the case you have to manually update the Makefile or set the `CUDA` environment variable before calling `make`.

To verify that the CUDA backend is working try:
```shell
python3 -c "import dsc; print(dsc.gpu.is_available() and dsc.gpu.is_cuda())"
```

### HIP backend
This provides GPU acceleration on AMD GPUs. To get started make sure to have the [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html#rocm-install-quick)
installed.

To build the C++ library with ROCm enabled simply specify `DSC_GPU=1`. ROCm will be detected automatically if you installed it.

**Note:** if you see errors when compiling with ROCm support make sure that the ROCm installation path specified in the Makefile
is correct. If this is not the case you have to manually update the Makefile or set the `ROCM` environment variable before calling `make`.

To verify that the ROCm backend is working try:
```shell
python3 -c "import dsc; print(dsc.gpu.is_available() and dsc.gpu.is_rocm())"
```

## Setting a default device
The default device in DSC is the CPU. This means that, if you don't specify anything, all the operations will be
performed on the CPU even when a GPU device is available. To set a different device as default you can use
```python
dsc.set_default_device('gpu')
```
This will make the GPU the default device and DSC will perform all the operations there by default.

## Running tests
DSC uses `pytest` to run unit tests against NumPy which is the reference for correctness.

The tests are structured as follows:
- `test_ops_common` and `test_indexing` are used to test operations both on CPU and GPU using NumPy as reference
- `test_ops_cpu` are CPU-specific
- `test_ops_gpu` are GPU-specific and they use PyTorch as reference

The device on which tests are run can be configured by setting the environment variable `DSC_DEVICE` before calling pytest.

**Note:** to use PyTorch with a ROCm-compatible GPU please refer to https://pytorch.org/get-started/locally/.

To run all the tests simple do:
```bash
cd python/tests/
pytest -s <test_file>.py --no-header --no-summary -q
```
**Note:** there are quite a few tests so to run them it's better to compile DSC with `DSC_FAST=1`.

## License
BSD-3-Clause
