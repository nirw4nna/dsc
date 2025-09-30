// Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#include <nanobind/nanobind.h>
// To convert to and from Python str
#include <nanobind/stl/string.h>
#include <sys/sysinfo.h>
#include <unordered_map>
#include "utils.h"
#include "dsc.h"


namespace nb = nanobind;
using namespace nb::literals;


// TODO: note that if there are concurrency issues this is probably the root cause (should be atomic!)

static dsc_ctx *g_ctx = nullptr;

namespace dsc::ctx {
void init(const usize mem_size) noexcept {
    g_ctx = dsc_ctx_init(mem_size);
}

void teardown() noexcept {
    if (g_ctx) dsc_ctx_free(g_ctx);
    g_ctx = nullptr;
}

dsc_ctx *get() noexcept {
    if (!g_ctx) {
        struct sysinfo info;
        DSC_ASSERT(sysinfo(&info) == 0);
        const usize total_ram = info.totalram * info.mem_unit;
        printf("DSC has not been explicitly initialized so it will try to reserve 80%% of the total available memory in the system.\n"
               "To explicitly initialize DSC call `dsc.init()` with a specific memory size in bytes.\n");
        init((usize) total_ram * 0.8);
    }
    return g_ctx;
}
}


void init_context(nb::module_& m) {
    nb::enum_<dsc_device_type>(m, "Device")
        .value("default", DEFAULT)
        .value("cpu", CPU)
        .value("gpu", GPU)
        .export_values();

    m.def("init", [](const usize mem_size) {
        dsc::ctx::init(mem_size);
    }, "nb"_a);

    // m.attr("_finalizer") = nb::capsule((void *) 1, "dsc_ctx", [](void *) noexcept {
    //    dsc::ctx::teardown();
    // });

    m.def("print_mem_usage", []() {
        dsc_print_mem_usage(dsc::ctx::get());
    });

    m.def("set_default_device", [](const Device &device) noexcept {
       dsc_set_default_device(dsc::ctx::get(), get_dsc_device(device));
    }, "device"_a);
}
