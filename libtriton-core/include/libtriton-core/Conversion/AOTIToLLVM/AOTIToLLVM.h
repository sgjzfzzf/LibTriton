#ifndef LIBTRITON_CORE_CONVERSION_AOTITOLLVM_AOTITOLLVM_H_
#define LIBTRITON_CORE_CONVERSION_AOTITOLLVM_AOTITOLLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace libtriton::aoti {

#define GEN_PASS_DECL_CONVERTAOTITOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTAOTITOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

void populateAOTIToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns);

void registerConvertAOTIToLLVMPass();
void registerConvertAOTIToLLVMInterface(mlir::DialectRegistry &registry);

} // namespace libtriton::aoti

#endif // LIBTRITON_CORE_CONVERSION_AOTITOLLVM_AOTITOLLVM_H_
