//  Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
//  All rights reserved.
//
//  This code is licensed under the terms of the 3-clause BSD license
//  (https://opensource.org/license/bsd-3-clause).

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/string.h>
#include "dsc.h"


using Device = std::variant<dsc_device_type, std::string>;

static DSC_INLINE dsc_device_type get_dsc_device(const Device &device) noexcept {
    if (std::holds_alternative<dsc_device_type>(device)) {
        return std::get<dsc_device_type>(device);
    }

    const auto &device_str = std::get<std::string>(device);
    if (device_str == "default") return DEFAULT;
    if (device_str == "cpu") return CPU;
    if (device_str == "gpu") return GPU;

    DSC_LOG_FATAL("unknown device %s", device_str.c_str());
}