#ifndef LIBTRITON_CORE_DIALECT_TRITONRT_TRANSFORMS_PASSES_H_
#define LIBTRITON_CORE_DIALECT_TRITONRT_TRANSFORMS_PASSES_H_

#include "libtriton-core/Dialect/TritonRT/IR/TritonRTDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"

namespace libtriton::triton_rt {

#define GEN_PASS_DECL
#include "libtriton-core/Dialect/TritonRT/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "libtriton-core/Dialect/TritonRT/Transforms/Passes.h.inc"

} // namespace libtriton::triton_rt

#endif // LIBTRITON_CORE_DIALECT_TRITONRT_TRANSFORMS_PASSES_H_

// Re-includable section: expand pass base class definitions.
#ifdef GEN_PASS_DEF_NORMALIZETRITONRTOPERANDS
namespace libtriton::triton_rt {
#include "libtriton-core/Dialect/TritonRT/Transforms/Passes.h.inc"
} // namespace libtriton::triton_rt
#undef GEN_PASS_DEF_NORMALIZETRITONRTOPERANDS
#endif
