# DSC Python Bindings
The generation of the Python bindings is handled by [nanobind](https://nanobind.readthedocs.io/en/latest/).

This folder is structured as follows:
- `CMakeLists.txt` is responsible for building the bindings. It's invoked directly by the root `Makefile`.
- `dsc.cpp` is the main entrypoint, it's responsible for setting up the module (name, version, ecc...)
- `<files>.cpp` all the other .cpp files define a piece of the API:
  - `context.cpp` exposes all the methods related to context management
  - `tensor.cpp` exposes all the tensor-related methods

New operations / data structures must be added in the proper .cpp file, if it doesn't exist yet the process is as follows:
1. Create the `.cpp` file with a function that has the signature `void init_XXX(nb::module_& m)`.
2. In `dsc.cpp` call that function (you have to forward-declare it)
3. Add the `.cpp` file to the `CMakeLists.txt`


## Stub Generation
To generate Python stubs with nanobind (useful for type-hints in IDEs) run:
```shell
python -m nanobind.stubgen -m dsc.core -O ./python/dsc
```

