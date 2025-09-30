// Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#include <nanobind/nanobind.h>

namespace nb = nanobind;

void init_context(nb::module_&);
void init_tensor(nb::module_&);

NB_MODULE(core, m) {
    init_context(m);
    init_tensor(m);
}