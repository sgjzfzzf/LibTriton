//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "libtriton-core/Conversion/TorchToLLVM/FuncBackendTypeConversion.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "libtriton-core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"

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
    target.addLegalDialect<mlir::BuiltinDialect, mlir::gpu::GPUDialect,
                           mlir::LLVM::LLVMDialect>();
    target.addIllegalDialect<mlir::cf::ControlFlowDialect,
                             mlir::torch::Torch::TorchDialect,
                             libtriton::tvm_ffi::TVMFFIDialect>();

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace libtriton::torch
