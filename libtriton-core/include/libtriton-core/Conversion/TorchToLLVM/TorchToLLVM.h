#ifndef LIBTRITON_CORE_CONVERSION_TORCHTOLLVM_TORCHTOLLVM_H_
#define LIBTRITON_CORE_CONVERSION_TORCHTOLLVM_TORCHTOLLVM_H_

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class LLVMTypeConverter;
} // namespace mlir

namespace libtriton::torch {

#define GEN_PASS_DECL_CONVERTTORCHTOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTTORCHTOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

void populateTorchToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns);

void registerConvertTorchToLLVMPass();

} // namespace libtriton::torch

#endif // LIBTRITON_CORE_CONVERSION_TORCHTOLLVM_TORCHTOLLVM_H_
