//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_PIPELINE_PIPELINE_H_
#define TRIDENT_CORE_CONVERSION_PIPELINE_PIPELINE_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace trident::torch {

#define GEN_PASS_DECL_TORCHTOLLVMPIPELINE
#include "trident-core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_TORCHTOLLVMPIPELINE
#include "trident-core/Conversion/Passes.h.inc"

} // namespace trident::torch

#endif // TRIDENT_CORE_CONVERSION_PIPELINE_PIPELINE_H_
