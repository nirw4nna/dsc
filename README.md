# DSC
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Unit Tests](https://github.com/dspcraft/dsc/actions/workflows/tests.yml/badge.svg)](https://github.com/dspcraft/dsc/actions/workflows/tests.yml)

DSC is an NumPy-compatible array framework for Python and C++.
The goal is to provide scientists and engineers with a familiar tool
that can be used during both the prototyping and implementation phase. 
The idea is to build something that can be as productive as NumPy to develop/test an algorithm 
but that can also be used to implement the same algorithm in production on systems where installing Python is not an option.

Some key features of DSC include:
- **Intuitive API**: DSC Python API closely resembles NumPy. DSC also has a fully 
functioning C++ API that is very similar to the Python API, users that know how to
use NumPy should be able to use also the C++ API without an extensive knowledge of C/C++.


- **No external dependencies**: DSC doesn't require external libraries to be efficient.
The core operations like FFTs are implemented from scratch in portable C++, this makes code
written using DSC extremely portable.


- **No runtime allocations**: DSC has it's own custom memory allocator, memory is pre-allocated
only once so no extra calls to `malloc()` or `free()` are required. It's also possible
to switch to a linear allocator to remove the (minimal) overhead introduced by a general purpose allocator.


- **Built-in tracing**: DSC can record traces that can be visualized using [Perfetto](https://ui.perfetto.dev/).
This feature can be enabled or disabled at compile time and is a core feature of DSC so it's naturally
available for both C++ and Python.


## Requirements
The only requirements are:
- A compiler with good support for C++20
- GNU Make for building

On a Linux-based system these can be obtained with:
```bash
sudo apt update
sudo apt install build-essential
```

## Installation
The recommended way to install DSC is from source:
```bash
git clone git@github.com:dspcraft/dsc.git
cd dsc/
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -e .
```

To build the C++ core:
```bash
make clean; make shared DSC_FAST=1
```
This will compile DSC without any debug information, you can specify different options
to enable/disable specific features:

| Option             | Description                                                                  |
|--------------------|------------------------------------------------------------------------------|
| DSC_DEBUG          | Compile with debug information and verbose logging (**default**)             |
| DSC_FAST           | Turn off logging and compile with the highest optimisation level             |
| DSC_CUDA           | Enable CUDA support (**default=OFF**)                                        |
| DSC_ENABLE_TRACING | Enable tracing for all operations                                            |
| DSC_MAX_TRACES     | Max number of traces that can be recorded (**default=1K**)                   |
| DSC_MAX_OBJS       | Max number of DSC tensors that can be used at the same time (**default=1K**) |

To verify that everything worked out as expected try a simple operation:
```bash
python3 -c "import dsc; x = dsc.arange(10); print(x)"
```

## Usage
The DSC Python API is very similar to NumPy but with a few key differences:
- **Initialization**: the memory pool that DSC can use must be explicitly initialized using `dsc.init(mem_size, scratch_size)`
at the beginning of your program. If this is not the case init will be called by the DSC Python wrapper, in this case the
amount of memory reserved for the main and the scratch memory will be 10% of the total memory of your system.


- **Tracing**: DSC has a built-in tracer. To use it in Python you can either call `dsc.start_recording()` and 
`dsc.stop_recording('traces.json')` before and after the code you want to trace or simply wrap your code inside a context manager
```python
with dsc.profile():
    # code here
```


- **NumPy interoperability**: DSC makes it easy to work with NumPy arrays. To create a `dsc.Tensor` from an `numpy.ndarray`
use `dsc.from_numpy(numpy_ndarray)` vice versa, to convert from a `dsc.Tensor` to a `numpy.ndarray` use `dsc_tensor.numpy()`.
Note that `numpy()` creates a view while `from_numpy()` creates a copy of the original array, it's not a good idea to frequently
switch between DSC and NumPy.

Everything else should be familiar to those used to work with NumPy.

To showcase some more features, let's implement an FFT-based filter in both Python and C++ using DSC.
### Python
```python
import dsc

def filterFFT(s: dsc.Tensor, b: dsc.Tensor) -> dsc.Tensor:
    # Determine the length of the output signal
    ls = len(s)
    lb = len(b)
    output_length = ls + lb - 1

    # Calculate the next power of two greater than or equal to the output length (for efficiency)
    fft_size = int(2 ** math.ceil(math.log2(output_length)))

    # Perform FFTs of the input signal and the filter coefficients
    S = dsc.rfft(s, n=fft_size)
    B = dsc.rfft(b, n=fft_size)

    # Multiply the FFTs to perform convolution in the frequency domain
    convolved_signal_freq = S * B

    # Perform inverse FFT to get back to the time domain
    filtered_signal = dsc.irfft(convolved_signal_freq)

    # Take the central part to have the same length as the input signal
    filtered_signal = filtered_signal[:output_length]

    return filtered_signal
```
### C++
```c++
#include "dsc_api.h"

dsc::tensor<float> filterFFT(const dsc::tensor<float>& s, const dsc::tensor<float>& b) {
    // Determine the length of the output signal
    const auto ls = s.size();
    const auto lb = b.size();
    const auto output_length = ls + lb - 1;
    
    // Calculate the next power of two greater than or equal to the output length (for efficiency)
    const auto fft_size = (int) (pow(2, ceil(log2((double) output_length))));
    
    // Perform FFTs of the input signal and the filter coefficients
    const auto S = dsc::rfft(s, fft_size);
    const auto B = dsc::rfft(b, fft_size);
    
    // Multiply the FFTs to perform convolution in the frequency domain
    const auto convolved_signal_freq = S * B;
    
    // Perform inverse FFT to get back to the time domain
    const auto filtered_signal = dsc::irfft(convolved_signal_freq);
    
    // Take the central part to have the same length as the input signal
    return filtered_signal.get(DSC_SLICE_TO(output_length));
}
```
As you can see both implementations look very similar and they also achieve similar performance as the core operations
are all implemented using the same low-level DSC core library.

## Performance
See [performance](benchmarks/perf.md) for a detailed comparison.

## Running Tests
DSC uses `pytest` to run unit tests against NumPy which is the reference for correctness.

To run all the tests simple do:
```bash
cd python/tests/
pytest -s test_ops.py --no-header --no-summary -q
```
**Note:** there are quite a few tests so to run them it's better to compile DSC with `DSC_FAST=1`. 

## License
BSD-3-Clause
