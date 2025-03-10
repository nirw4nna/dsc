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


## Profiling
I need a mechanism to profile DSC code. The key points to keep in mind are:
- Python overhead can be ignored for kernels that are dispatched directly (ie. matmul: here
most of the time is spent in C++, the Python overhead is only a few us)
- Kernels that are defined in Python (ie. gelu, var, LayerNorm) must be profiled
- Traces must be available in Python
- Tracing must be opt-in, both in Python (using env vars?) and in C++ (compile-time flags?)

One possibility is to use C++ to generate also the Python traces but in this case it's hard to capture the context
(what parameters should we print? what metrics?).

Another possibility is to use the C++ portion to collect traces but then expose those traces via the standard low-level
API so that all the manipulations like sorting, grouping, filtering ecc... can be done in Python.
Then, it would be great if I could 'group' events together. For example:
```python
def softmax(x: Tensor, axis: int = -1) -> Tensor:
    e = exp((x - max(x, axis=axis, keepdims=True)))
    sum_e = sum(e, axis=axis, keepdims=True)
    return e / sum_e
```
This code will generate a bunch of events (a reduction along an axis, a binary op, lots of allocations...), I need a mechanism
to mark them as one big meta-operation, this way I know when looking at the traces where they come from.
Same goes for the forward step of an `nn.Module`.

An idea could be marking the 'meta-functions' with some decorators like:
```python3
@trace('Softmax')
def softmax(...):
   ...
```
Here the decorator basically creates a new trace (ask the C++ code to generate the trace?) before and after the function
call.

**Questions:**
- How does grouping (ie. an event that spans multiple, smaller, events) work in chrome-traces/Perfetto?
- What's the latency of this decorator mechanism?
- How can we exchange events efficiently between Python and C++?