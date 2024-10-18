# DSC

## Installation
The recommended way to install dsc is from source.

```bash
git clone git@github.com:dspcraft/dsc.git
cd dsc/
pip install -e .
```

Next, build the C++ core via `Makefile`.

```bash
make clean; make shared
```
You can specify different options during compilation to enable/disable specific features:

| Option             | Description                                                      |
|--------------------|------------------------------------------------------------------|
| DSC_DEBUG          | Compile with debug information and verbose logging (**default**) |
| DSC_FAST           | Turn off logging and compile with the highest optimisation level |
| DSC_ENABLE_TRACING | Enable tracing for all operations                                |
| DSC_MAX_FFT_PLANS  | Max number of FFT plans that can be cached (**default=16**)      |
| DSC_MAX_TRACES     | Max number of traces that can be recorder (**default=1K**)       |


## Running Tests
```bash
cd python/tests/
pytest -s test_ops.py --no-header --no-summary -q
```