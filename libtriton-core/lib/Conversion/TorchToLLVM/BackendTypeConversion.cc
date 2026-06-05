//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "libtriton-core/Conversion/TorchToLLVM/BackendTypeConversion.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

using namespace mlir;
using namespace mlir::torch;

//===----------------------------------------------------------------------===//
// Type conversion setup.
//===----------------------------------------------------------------------===//

/// Convert Torch tensor types (ValueTensorType, NonValueTensorType) to
/// LLVM pointer type.
static void
setupTorchTensorToLLVMPtrConversion(LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [](Torch::BaseTensorType type) -> std::optional<Type> {
        return LLVM::LLVMPointerType::get(type.getContext());
      });
  typeConverter.addTargetMaterialization(
      [](OpBuilder &builder, LLVM::LLVMPointerType type, ValueRange inputs,
         Location loc) -> Value {
        return UnrealizedConversionCastOp::create(builder, loc, TypeRange(type),
                                                  inputs)
            .getResult(0);
      });
  typeConverter.addSourceMaterialization(
      [](OpBuilder &builder, Torch::BaseTensorType type, ValueRange inputs,
         Location loc) -> Value {
        return UnrealizedConversionCastOp::create(builder, loc, TypeRange(type),
                                                  inputs)
            .getResult(0);
      });
}

/// Convert Torch BoolType to builtin i1.
static void setupTorchBoolToI1Conversion(LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion([](Torch::BoolType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 1);
  });
  typeConverter.addTargetMaterialization([](OpBuilder &builder,
                                            IntegerType type, ValueRange inputs,
                                            Location loc) -> Value {
    return UnrealizedConversionCastOp::create(builder, loc, TypeRange(type),
                                              inputs)
        .getResult(0);
  });
  typeConverter.addSourceMaterialization(
      [](OpBuilder &builder, Torch::BoolType type, ValueRange inputs,
         Location loc) -> Value {
        return UnrealizedConversionCastOp::create(builder, loc, TypeRange(type),
                                                  inputs)
            .getResult(0);
      });
}

/// Convert Torch IntType to builtin i64.
static void setupTorchIntToI64Conversion(LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion([](Torch::IntType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 64);
  });
  typeConverter.addTargetMaterialization([](OpBuilder &builder,
                                            IntegerType type, ValueRange inputs,
                                            Location loc) -> Value {
    return UnrealizedConversionCastOp::create(builder, loc, TypeRange(type),
                                              inputs)
        .getResult(0);
  });
  typeConverter.addSourceMaterialization(
      [](OpBuilder &builder, Torch::IntType type, ValueRange inputs,
         Location loc) -> Value {
        return UnrealizedConversionCastOp::create(builder, loc, TypeRange(type),
                                                  inputs)
            .getResult(0);
      });
}

/// Convert Torch FloatType to builtin f64.
static void setupTorchFloatToF64Conversion(LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion([](Torch::FloatType type) -> std::optional<Type> {
    return Float64Type::get(type.getContext());
  });
  typeConverter.addTargetMaterialization([](OpBuilder &builder,
                                            Float64Type type, ValueRange inputs,
                                            Location loc) -> Value {
    return UnrealizedConversionCastOp::create(builder, loc, TypeRange(type),
                                              inputs)
        .getResult(0);
  });
  typeConverter.addSourceMaterialization(
      [](OpBuilder &builder, Torch::FloatType type, ValueRange inputs,
         Location loc) -> Value {
        return UnrealizedConversionCastOp::create(builder, loc, TypeRange(type),
                                                  inputs)
            .getResult(0);
      });
}

void libtriton::torch::setupBackendTypeConversion(
    ConversionTarget &target, LLVMTypeConverter &typeConverter) {
  setupTorchTensorToLLVMPtrConversion(typeConverter);
  setupTorchBoolToI1Conversion(typeConverter);
  setupTorchIntToI64Conversion(typeConverter);
  setupTorchFloatToF64Conversion(typeConverter);
}
