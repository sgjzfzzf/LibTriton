#include "libtriton-core/Dialect/TorchExt/Transforms/RewriteTorchAsTorchExt.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtOps.h"

#include <regex>

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"

namespace libtriton::torchext {

#define GEN_PASS_DEF_REWRITETORCHASTORCHEXT
#include "libtriton-core/Dialect/TorchExt/Transforms/Passes.h.inc"

namespace {

/// Rewrites any Torch dialect op whose name starts with "aten." into a
/// TorchExt torchext.call_dispatcher op. The "aten." prefix is replaced
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

    // Match "torch.aten.<OpName>[.<Overload>]" using std::regex (ECMAScript),
    // which supports (?:...) non-capturing groups for the optional overload.
    static const std::regex re(
        R"(^torch\.aten\.([_a-zA-Z]+)(?:\.([_a-zA-Z]+))?$)");
    std::smatch m;
    std::string opNameCopy = opName.str();
    if (!std::regex_match(opNameCopy, m, re)) {
      return mlir::failure();
    }

    // m[1] = base op name (e.g. "add"), m[2] = overload name (e.g. "Tensor").
    const std::string opNameStr = "aten::" + m[1].str();
    const std::string overloadNameStr = m[2].str();

    rewriter.replaceOpWithNewOp<libtriton::torchext::CallDispatcherOp>(
        op, op->getResultTypes(), rewriter.getStringAttr(opNameStr),
        rewriter.getStringAttr(overloadNameStr), op->getOperands());
    return mlir::success();
  }
};

class RewriteTorchAsTorchExtPass
    : public libtriton::torchext::impl::RewriteTorchAsTorchExtBase<
          RewriteTorchAsTorchExtPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::RewritePatternSet patterns(&context);
    populateRewriteTorchAsTorchExtPatterns(patterns);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateRewriteTorchAsTorchExtPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertAtenOp>(patterns.getContext());
}

} // namespace libtriton::torchext
