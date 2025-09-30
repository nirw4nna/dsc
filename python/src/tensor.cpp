// Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#include "context.h"
#include "dsc.h"
#include "dsc_device.h"
#include "utils.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/variant.h>


namespace nb = nanobind;
using namespace nb::literals;

using NDarray = nb::ndarray<nb::ro, nb::c_contig>;
using Tensor = std::variant<dsc_tensor*, NDarray>;
using Shape = std::variant<int, std::vector<int>, Tensor>;
using Scalar = std::variant<bool, int, f64>;


static DSC_INLINE dsc_dtype dtype_to_dsc(const nb::dlpack::dtype& dt) noexcept {
    if (dt == nb::dtype<bool>()) return BOOL;
    if (dt == nb::dtype<i32>()) return I32;
    if (dt == nb::dtype<f32>()) return F32;
    if (dt == nb::dtype<f64>()) return F64;

    DSC_LOG_FATAL("unknown dlpack dtype %d", dt.code);
}

static dsc_tensor *create_tensor(const Shape &shape, const dsc_dtype dtype,
                                 const dsc_device_type device,
                                 const void *data = nullptr,
                                 const dsc_device_type data_device = DEFAULT) noexcept {
    int shape_arr[DSC_MAX_DIMS];
    int n_dim = 0;
    if (std::holds_alternative<int>(shape)) {
        shape_arr[n_dim++] = std::get<int>(shape);
    } else if (std::holds_alternative<std::vector<int>>(shape)) {
        for (const auto d : shape_arr) {
            shape_arr[n_dim++] = d;
        }
    } else {
        if (const auto x = std::get<Tensor>(shape); std::holds_alternative<NDarray>(x)) {
            const auto x_ = std::get<NDarray>(x);
            for (int i = 0; i < x_.ndim(); ++i) shape_arr[n_dim++] = (int) x_.shape(i);
        } else {
            const auto x_ = std::get<dsc_tensor *>(x);
            for (int i = 0; i < x_->n_dim; ++i) shape_arr[n_dim++] = x_->shape[i];
        }
    }
    return dsc_new_tensor(dsc::ctx::get(), n_dim, shape_arr, dtype, device, nullptr, false, data, data_device);
}

static DSC_INLINE dsc_dtype dtype_or_default(const std::optional<Tensor> &x,
                                             const std::optional<dsc_dtype> dtype) {
    if (dtype.has_value()) {
        return dtype.value();
    }

    DSC_ASSERT(x.has_value());
    const auto& x_ = x.value();

    dsc_dtype d_;
    if (std::holds_alternative<NDarray>(x_)) {
        const auto x_nd = std::get<NDarray>(x_);
        d_ = dtype_to_dsc(x_nd.dtype());
    } else {
        const auto x_dsc = std::get<dsc_tensor *>(x_);
        d_ = x_dsc->dtype;
    }
    return d_;
}

void init_tensor(nb::module_& m) {
    nb::enum_<dsc_dtype>(m, "Dtype", nb::is_arithmetic())
        .value("bool_", BOOL)
        .value("i32", I32)
        .value("bf16", BF16)
        .value("f32", F32)
        .value("f64", F64)
        .export_values();

    nb::class_<dsc_tensor>(m, "Tensor")
        .def_prop_ro("dtype", [](const dsc_tensor *x) noexcept { return x->dtype; });



    m.def("from_numpy", [](const NDarray& x, const Device &device) {
        // TODO: this should work only on CPU!
        const dsc_dtype out_dtype = dtype_to_dsc(x.dtype());
        const dsc_device_type out_device = get_dsc_device(device);
        return create_tensor(x, out_dtype, out_device, x.data(), CPU);
    }, "x"_a, "device"_a = DEFAULT);

    m.def("frombuffer", [](const Shape &shape, const dsc_dtype dtype, const void *data, const Device &device, const Device &data_device) {
        return create_tensor(shape, dtype, get_dsc_device(device), data, get_dsc_device(data_device));
    }, "shape"_a, "dtype"_a, "data"_a, "device"_a = DEFAULT, "data_device"_a = CPU);

    m.def("empty", [](const Shape &shape, const dsc_dtype dtype,
                      const Device &device) noexcept {
        return create_tensor(shape, dtype, get_dsc_device(device));
    }, "shape"_a, "dtype"_a = F32, "device"_a = DEFAULT);

    m.def("empty_like", [](const Tensor &x, const std::optional<dsc_dtype> dtype,
                      const Device &device) noexcept {
        return create_tensor(x, dtype_or_default(x, dtype), get_dsc_device(device));
    }, "x"_a, "dtype"_a = nb::none(), "device"_a = DEFAULT);

    m.def("full", [](const Shape &shape, const Scalar fill_value,
                     const dsc_dtype dtype, const Device &device) noexcept {
        const auto out = create_tensor(shape, dtype, get_dsc_device(device));
        // TODO: fill func
        return out;
    }, "shape"_a, "fill_value"_a, "dtype"_a = F32, "device"_a = DEFAULT);

    m.def("full_like", [](const Tensor &x, const Scalar fill_value,
                     const std::optional<dsc_dtype> dtype, const Device &device) noexcept {
        const auto out = create_tensor(x, dtype_or_default(x, dtype), get_dsc_device(device));
        // TODO: fill func
        return out;
    }, "x"_a, "fill_value"_a, "dtype"_a = nb::none(), "device"_a = DEFAULT);

    m.def("zeroes", [](const Shape &shape, const dsc_dtype dtype,
                       const Device &device) noexcept {
        const auto device_ = get_dsc_device(device);
        const auto out = create_tensor(shape, dtype, device_);
        const dsc_ctx *ctx = dsc::ctx::get();
        DSC_DATA(void, out);
        dsc_get_device(device_)->memset(out_data, 0, dsc_tensor_nbytes(out));
        return out;
    }, "shape"_a, nb::kw_only(), "dtype"_a = F32, "device"_a = DEFAULT);

    m.def("zeroes_like", [](const Tensor &x, const std::optional<dsc_dtype> dtype,
                            const Device &device) noexcept {
        const auto device_ = get_dsc_device(device);
        const auto out = create_tensor(x, dtype_or_default(x, dtype), device_);
        const dsc_ctx *ctx = dsc::ctx::get();
        DSC_DATA(void, out);
        dsc_get_device(device_)->memset(out_data, 0, dsc_tensor_nbytes(out));
        return out;
    }, "x"_a, "dtype"_a = nb::none(), "device"_a = DEFAULT);

    m.def("ones", [](const Shape &shape, const dsc_dtype dtype,
                     const Device &device) noexcept {
        const auto out = create_tensor(shape, dtype, get_dsc_device(device));
        // TODO: fill func
        return out;
    }, "shape"_a, "dtype"_a = F32, "device"_a = DEFAULT);

    m.def("ones_like", [](const Tensor &x, const std::optional<dsc_dtype> dtype,
                          const Device &device) noexcept {
        const auto out = create_tensor(x, dtype_or_default(x, dtype), get_dsc_device(device));
        // TODO: fill func
        return out;
    }, "shape"_a, "dtype"_a = nb::none(), "device"_a = DEFAULT);

}