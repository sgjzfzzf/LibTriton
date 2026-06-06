#include "libtriton-core/Dialect/AOTInductor/Transforms/RewriteTorchAsAOTI.h"
#include "libtriton-core/Dialect/AOTInductor/IR/AOTInductorDialect.h"
#include "libtriton-core/Dialect/AOTInductor/IR/AOTInductorOps.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"

namespace libtriton::aoti {

#define GEN_PASS_DEF_REWRITETORCHASAOTI
#include "libtriton-core/Dialect/AOTInductor/Transforms/Passes.h.inc"

namespace {

/// Rewrites any Torch dialect op whose name starts with "aten." into an
/// AOTInductor aoti.torch_call_dispatcher op. The "aten." prefix is replaced
/// with "aten::" as the dispatcher op_name, and the overload_name is empty.
struct ConvertAtenOp : public mlir::RewritePattern {
  ConvertAtenOp(mlir::MLIRContext *context)
      : mlir::RewritePattern(mlir::Pattern::MatchAnyOpTypeTag(),
                             /*benefit=*/1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!llvm::isa<mlir::torch::Torch::TorchDialect>(op->getDialect())) {
      return mlir::failure();
    }

    llvm::StringRef opName = op->getName().getStringRef();
    constexpr llvm::StringRef kAtenPrefix = "torch.aten.";
    if (!opName.starts_with(kAtenPrefix)) {
      return mlir::failure();
    }

    // Build op_name: strip "torch." prefix, then replace "aten." with "aten::"
    llvm::Twine newOpName = "aten::" + opName.drop_front(kAtenPrefix.size());

    rewriter.replaceOpWithNewOp<libtriton::aoti::TorchCallDispatcherOp>(
        op, op->getResultTypes(), rewriter.getStringAttr(newOpName),
        rewriter.getStringAttr(""), op->getOperands());
    return mlir::success();
  }
};

class RewriteTorchAsAOTIPass
    : public impl::RewriteTorchAsAOTIBase<RewriteTorchAsAOTIPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::RewritePatternSet patterns(&context);
    populateRewriteTorchAsAOTIPatterns(patterns);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateRewriteTorchAsAOTIPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertAtenOp>(patterns.getContext());
}

} // namespace libtriton::aoti
