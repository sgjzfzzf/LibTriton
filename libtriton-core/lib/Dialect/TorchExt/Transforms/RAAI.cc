//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/Pass/Pass.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

namespace libtriton::torch {

#define GEN_PASS_DEF_RAAI
#define GEN_PASS_REGISTRATION_RAAI
#include "libtriton-core/Dialect/TorchExt/Transforms/Passes.h.inc"

namespace {

class RAAIPass : public impl::RAAIBase<RAAIPass> {
public:
  void runOnOperation() final {
    mlir::Region &body = getOperation().getBody();
    // Only handle single-block regions to keep the analysis simple.
    if (!body.hasOneBlock()) {
      return;
    }
    mlir::Block &block = body.front();

    // --- Collect allocated values ---
    llvm::SmallVector<mlir::Value> allocatedVals;
    block.walk([&](mlir::Operation *op) {
      // torch.prim.ListConstruct → List allocation
      if (mlir::torch::Torch::PrimListConstructOp listCtor =
              llvm::dyn_cast<mlir::torch::Torch::PrimListConstructOp>(op)) {
        allocatedVals.push_back(listCtor.getResult());
      }
      // torch.aten.* → Tensor/list allocation (only tensor/list-typed results)
      else if (mlir::OperationName opName = op->getName();
               opName.getDialectNamespace() == "torch" &&
               opName.getStringRef().starts_with("torch.aten.")) {
        for (mlir::Value result :
             llvm::make_filter_range(op->getResults(), [](mlir::Value val) {
               return llvm::isa<mlir::torch::Torch::ValueTensorType,
                                mlir::torch::Torch::NonValueTensorType,
                                mlir::torch::Torch::ListType>(val.getType());
             })) {
          allocatedVals.push_back(result);
        }
      }
    });

    // --- Insert ref-count ops before the terminator ---
    mlir::Operation *terminator = block.getTerminator();
    mlir::OpBuilder builder(terminator);
    mlir::Location loc = terminator->getLoc();

    // Values yielded (returned) from this block are the terminator's operands.
    mlir::ValueRange yieldedVals = terminator->getOperands();

    // --- IncRef for every yielded value (allocated or external arg) ---
    for (mlir::Value val : yieldedVals) {
      libtriton::torchext::ObjectIncRefOp::create(builder, loc, val);
    }

    // --- DecRef for every internally-allocated value ---
    for (mlir::Value val : allocatedVals) {
      libtriton::torchext::ObjectDecRefOp::create(builder, loc, val);
    }
  }
};

} // namespace

} // namespace libtriton::torch
