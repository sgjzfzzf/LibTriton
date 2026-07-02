#ifndef TRIDENT_CORE_CONVERSION_TORCHTOLLVM_TORCHTOLLVM_H_
#define TRIDENT_CORE_CONVERSION_TORCHTOLLVM_TORCHTOLLVM_H_

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class LLVMTypeConverter;
} // namespace mlir

namespace trident::torch {

#define GEN_PASS_DECL_CONVERTTORCHTOLLVM
#include "trident-core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTTORCHTOLLVM
#include "trident-core/Conversion/Passes.h.inc"

void populateTorchToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns);

void registerConvertTorchToLLVMPass();
void registerConvertTorchToLLVMInterface(mlir::DialectRegistry &registry);

} // namespace trident::torch

#endif // TRIDENT_CORE_CONVERSION_TORCHTOLLVM_TORCHTOLLVM_H_
