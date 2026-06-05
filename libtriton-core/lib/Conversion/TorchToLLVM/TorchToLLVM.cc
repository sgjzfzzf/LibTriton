#include "libtriton-core/Conversion/TorchToLLVM/TorchToLLVM.h"
#include "libtriton-core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"

namespace libtriton::torch {

#define GEN_PASS_DEF_CONVERTTORCHTOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

namespace {

class ConvertTorchToLLVMPass
    : public impl::ConvertTorchToLLVMBase<ConvertTorchToLLVMPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::LLVMTypeConverter typeConverter(&context);
    mlir::ConversionTarget target(context);
    mlir::RewritePatternSet patterns(&context);
    populateTorchToLLVMConversionPatterns(target, typeConverter, patterns);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct TorchToLLVMDialectInterface
    : public mlir::ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(
      mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
      mlir::RewritePatternSet &patterns) const final {
    populateTorchToLLVMConversionPatterns(target, typeConverter, patterns);
  }
};

} // namespace

void populateTorchToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  // Set up type conversions (Tensor -> llvm.ptr, Bool -> i1, Int -> i64,
  // Float -> f64). Generator conversion is not implemented yet.
  setupBackendTypeConversion(target, typeConverter);

  // TODO: Add op conversion patterns for Torch dialect ops.
  target.addIllegalDialect<mlir::torch::Torch::TorchDialect>();
  target.addLegalDialect<mlir::BuiltinDialect, mlir::LLVM::LLVMDialect>();
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
}

void registerConvertTorchToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, mlir::torch::Torch::TorchDialect *dialect) {
        dialect->addInterfaces<TorchToLLVMDialectInterface>();
      });
}

} // namespace libtriton::torch
