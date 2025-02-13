## Code Gen
I'm thinking about adding some table-driven code generation into dsc.
Right now, if you want to add a new operation you have to go through the following steps:

1. Declare the API function in `dsc.h`
2. Implement the function (parameter validation + dispatching) in `dsc.cpp`
3. For each backend:
   1. Declare the function that actually implements the core logic in `dsc_xxx.h`
   2. Actually implement the core logic of the function in `dsc_xxx.cpp`
4. Implement the binding in `_bindings.py`

The actual logic itself for the *majority* of operations is basically the same.
Consider a generic binary operation, the flow is almost always:
1. Validation + dispatch
```c++
dsc_tensor *dsc_add(dsc_ctx *ctx,
                    dsc_tensor *xa,
                    dsc_tensor *xb,
                    dsc_tensor *out) {
    validate_binary_params();

    DSC_DISPATCH(xa->device, add, xa, xb, out);

    cleanup_binary();

    return out;
}
```
2. Implementation
```c++
void dsc_cpu_add(dsc_device *,
                 const dsc_tensor *xa,
                 const dsc_tensor *xb,
                 dsc_tensor *out) {
    binary_op(xa, xb, out, cpu_add_op());
}
```
The same goes for unary operations and for reductions. The only exceptions at the moment is probably
the GEMM and the operations related to indexing and slicing.

Also, most of the code in `dsc_ops.h` can be generate trivially.

And then there are the aspects related to testing and benchmarking: using this kind of approach could also
lead to better testing and benchmarking (ie. when I specify a new operation I can also specify 'how it should be tested / benchmarked').

Another important point: with this approach it could be even easier to implement tracing. For example, 
if I want to use the current approach I can just define the parameters that are important and generate
the macros and stuff.


**Key things to keep in mind:**
- The generated code must live alongside the handwritten code, I don't have a clear idea on how to do this,
same goes for the bindings. For tests and benchmarks is easier because I can actually put them in multiple files
right away.
- (this is related to the previous point) Limit file proliferation. I don't want the number of files to explode
due to this feature.
- **Generated code must be versioned** (ie. pushed to GitHub) **and must be modifiable** and the code generator
must not erase these updates if it's re-run.