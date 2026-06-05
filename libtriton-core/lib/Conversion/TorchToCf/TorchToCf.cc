#include "libtriton-core/Conversion/TorchToCf/TorchToCf.h"
#include "libtriton-core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

namespace libtriton::torch {

#define GEN_PASS_DEF_CONVERTTORCHTOCF
#include "libtriton-core/Conversion/Passes.h.inc"

namespace {

class ConvertRuntimeAssertOp
    : public mlir::OpConversionPattern<mlir::torch::Torch::RuntimeAssertOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::RuntimeAssertOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::cf::AssertOp>(op, adaptor.getCondition(),
                                                    adaptor.getMessage());
    return mlir::success();
  }
};

class ConvertTorchToCfPass
    : public impl::ConvertTorchToCfBase<ConvertTorchToCfPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::LLVMTypeConverter typeConverter(&context);
    mlir::ConversionTarget target(context);
    mlir::RewritePatternSet patterns(&context);
    setupBackendTypeConversion(target, typeConverter);
    populateTorchToCfConversionPatterns(target, typeConverter, patterns);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateTorchToCfConversionPatterns(mlir::ConversionTarget &target,
                                         mlir::TypeConverter &typeConverter,
                                         mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertRuntimeAssertOp>(typeConverter, patterns.getContext());
  target.addIllegalOp<mlir::torch::Torch::RuntimeAssertOp>();
  target.addLegalDialect<mlir::cf::ControlFlowDialect>();
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
}

} // namespace libtriton::torch
