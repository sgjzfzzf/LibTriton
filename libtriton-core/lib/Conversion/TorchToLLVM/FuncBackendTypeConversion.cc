//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "libtriton-core/Conversion/TorchToLLVM/FuncBackendTypeConversion.h"
#include "libtriton-core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace libtriton::torch {

#define GEN_PASS_DEF_FUNCBACKENDTYPECONVERSION
#include "libtriton-core/Conversion/Passes.h.inc"

namespace {

class FuncBackendTypeConversionPass
    : public impl::FuncBackendTypeConversionBase<
          FuncBackendTypeConversionPass> {
public:
  void runOnOperation() final {
    mlir::LLVMTypeConverter typeConverter(&getContext());
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());

    setupBackendTypeConversion(target, typeConverter);
    populateFuncBackendTypeConversionPatterns(typeConverter, patterns, target);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace libtriton::torch
