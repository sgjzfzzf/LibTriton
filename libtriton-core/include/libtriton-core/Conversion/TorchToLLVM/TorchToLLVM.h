#ifndef LIBTRITON_CORE_CONVERSION_TORCHTOLLVM_TORCHTOLLVM_H_
#define LIBTRITON_CORE_CONVERSION_TORCHTOLLVM_TORCHTOLLVM_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace libtriton::torch {

#define GEN_PASS_DECL_TORCHTOLLVMPIPELINE
#include "libtriton-core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_TORCHTOLLVMPIPELINE
#include "libtriton-core/Conversion/Passes.h.inc"

} // namespace libtriton::torch

#endif // LIBTRITON_CORE_CONVERSION_TORCHTOLLVM_TORCHTOLLVM_H_
