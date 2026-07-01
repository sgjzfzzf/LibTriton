//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBTRITON_CORE_DIALECT_TORCHEXT_TRANSFORMS_RAAI_H_
#define LIBTRITON_CORE_DIALECT_TORCHEXT_TRANSFORMS_RAAI_H_

#include "mlir/Pass/Pass.h"

namespace libtriton::torch {

#define GEN_PASS_DECL_RAAI
#include "libtriton-core/Dialect/TorchExt/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION_RAAI
#include "libtriton-core/Dialect/TorchExt/Transforms/Passes.h.inc"

} // namespace libtriton::torch

#endif // LIBTRITON_CORE_DIALECT_TORCHEXT_TRANSFORMS_RAAI_H_
