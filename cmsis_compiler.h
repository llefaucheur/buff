#ifndef CMSIS_COMPILER_H
#define CMSIS_COMPILER_H

/*
 * Flat-package shim for CMSIS-DSP headers.
 *
 * The upstream CMSIS-DSP arm_math_types.h includes cmsis_compiler.h from
 * CMSIS-Core when building with GCC/clang. This package is intentionally flat,
 * so route those compiler macro definitions to the local compatibility header.
 */

#include "cmsis_flat_compat.h"

#endif
