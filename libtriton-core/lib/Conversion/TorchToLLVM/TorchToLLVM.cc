#include "libtriton-core/Conversion/TorchToLLVM/TorchToLLVM.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace libtriton::torch {

#define GEN_PASS_DEF_TORCHTOLLVMPIPELINE
#include "libtriton-core/Conversion/Passes.h.inc"

namespace {

class TorchToLLVMPipelinePass
    : public impl::TorchToLLVMPipelineBase<TorchToLLVMPipelinePass> {
public:
  void runOnOperation() final {
    // TODO: Implement the Torch → LLVM pipeline.
  }
};

} // namespace

} // namespace libtriton::torch
