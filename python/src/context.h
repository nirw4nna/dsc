#pragma once

#include "dsc.h"

namespace dsc::ctx {

extern void init(usize mem_size) noexcept;

extern void teardown() noexcept;

extern dsc_ctx *get() noexcept;

}
