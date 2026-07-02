//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_TVMFFITOLLVM_TVMFFITOLLVM_H_
#define TRIDENT_CORE_CONVERSION_TVMFFITOLLVM_TVMFFITOLLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace trident::tvm_ffi {

#define GEN_PASS_DECL_CONVERTTVMFFITOLLVM
#include "trident-core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTTVMFFITOLLVM
#include "trident-core/Conversion/Passes.h.inc"

void populateTVMFFIToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns);

void registerConvertTVMFFIToLLVMPass();
void registerConvertTVMFFIToLLVMInterface(mlir::DialectRegistry &registry);

} // namespace trident::tvm_ffi

#endif // TRIDENT_CORE_CONVERSION_TVMFFITOLLVM_TVMFFITOLLVM_H_
