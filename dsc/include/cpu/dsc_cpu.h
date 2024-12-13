// Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "dsc.h"


struct dsc_device;


// ============================================================
// CPU-specific operations
//

// extern void dsc_cpu_arange(dsc_device *, dsc_tensor *DSC_RESTRICT x);

extern void dsc_cpu_randn(dsc_device *, dsc_tensor *DSC_RESTRICT x);
