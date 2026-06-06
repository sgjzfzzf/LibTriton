//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "libtriton-core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
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

/// Convert Torch OptionalType to an LLVM struct with an i1 tag + contained
/// type. The i1 field indicates whether the optional is None (false) or has a
/// value (true).
static void
setupTorchOptionalToLLVMStructConversion(LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [&typeConverter](Torch::OptionalType type) -> std::optional<Type> {
        MLIRContext *ctx = type.getContext();
        Type containedType = type.getContainedType();
        Type convertedContained = typeConverter.convertType(containedType);
        return LLVM::LLVMStructType::getLiteral(
            ctx, {IntegerType::get(ctx, 1), convertedContained});
      });
  typeConverter.addTargetMaterialization(
      [](OpBuilder &builder, LLVM::LLVMStructType type, ValueRange inputs,
         Location loc) -> Value {
        return UnrealizedConversionCastOp::create(builder, loc, TypeRange(type),
                                                  inputs)
            .getResult(0);
      });
  typeConverter.addSourceMaterialization(
      [](OpBuilder &builder, Torch::OptionalType type, ValueRange inputs,
         Location loc) -> Value {
        return UnrealizedConversionCastOp::create(builder, loc, TypeRange(type),
                                                  inputs)
            .getResult(0);
      });
}

/// Convert Torch ListType to LLVM pointer type.
static void
setupTorchListToLLVMPtrConversion(LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion([](Torch::ListType type) -> std::optional<Type> {
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
      [](OpBuilder &builder, Torch::ListType type, ValueRange inputs,
         Location loc) -> Value {
        return UnrealizedConversionCastOp::create(builder, loc, TypeRange(type),
                                                  inputs)
            .getResult(0);
      });
}

/// Convert Torch TupleType to an LLVM struct with each contained type
/// converted via the type converter.
static void
setupTorchTupleToLLVMStructConversion(LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [&typeConverter](Torch::TupleType type) -> std::optional<Type> {
        MLIRContext *ctx = type.getContext();
        SmallVector<Type> convertedTypes;
        for (Type contained : type.getContainedTypes()) {
          Type converted = typeConverter.convertType(contained);
          if (!converted)
            return std::nullopt;
          convertedTypes.push_back(converted);
        }
        return LLVM::LLVMStructType::getLiteral(ctx, convertedTypes);
      });
  typeConverter.addTargetMaterialization(
      [](OpBuilder &builder, LLVM::LLVMStructType type, ValueRange inputs,
         Location loc) -> Value {
        return UnrealizedConversionCastOp::create(builder, loc, TypeRange(type),
                                                  inputs)
            .getResult(0);
      });
  typeConverter.addSourceMaterialization(
      [](OpBuilder &builder, Torch::TupleType type, ValueRange inputs,
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

/// Convert Torch DeviceType to an LLVM struct<i32, i32>.
/// The first i32 represents the device type (e.g., CPU=0, CUDA=1),
/// and the second i32 represents the device index.
static void
setupTorchDeviceToLLVMStructConversion(LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [](Torch::DeviceType type) -> std::optional<Type> {
        MLIRContext *ctx = type.getContext();
        return LLVM::LLVMStructType::getLiteral(
            ctx, {IntegerType::get(ctx, 32), IntegerType::get(ctx, 32)});
      });
  typeConverter.addTargetMaterialization(
      [](OpBuilder &builder, LLVM::LLVMStructType type, ValueRange inputs,
         Location loc) -> Value {
        return UnrealizedConversionCastOp::create(builder, loc, TypeRange(type),
                                                  inputs)
            .getResult(0);
      });
  typeConverter.addSourceMaterialization(
      [](OpBuilder &builder, Torch::DeviceType type, ValueRange inputs,
         Location loc) -> Value {
        return UnrealizedConversionCastOp::create(builder, loc, TypeRange(type),
                                                  inputs)
            .getResult(0);
      });
}

/// Convert Torch StringType to LLVM pointer type.
static void
setupTorchStringToLLVMPtrConversion(LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [](Torch::StringType type) -> std::optional<Type> {
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
      [](OpBuilder &builder, Torch::StringType type, ValueRange inputs,
         Location loc) -> Value {
        return UnrealizedConversionCastOp::create(builder, loc, TypeRange(type),
                                                  inputs)
            .getResult(0);
      });
}

void libtriton::torch::populateFuncBackendTypeConversionPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType());
  });
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  target.addDynamicallyLegalOp<func::CallOp>(
      [&](func::CallOp op) { return typeConverter.isLegal(op); });

  cf::populateCFStructuralTypeConversionsAndLegality(typeConverter, patterns,
                                                     target);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
  target.addLegalOp<ModuleOp>();

  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
           isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                            typeConverter) ||
           isLegalForReturnOpTypeConversionPattern(op, typeConverter);
  });
}

void libtriton::torch::setupBackendTypeConversion(
    ConversionTarget &target, LLVMTypeConverter &typeConverter) {
  setupTorchTensorToLLVMPtrConversion(typeConverter);
  setupTorchOptionalToLLVMStructConversion(typeConverter);
  setupTorchListToLLVMPtrConversion(typeConverter);
  setupTorchTupleToLLVMStructConversion(typeConverter);
  setupTorchBoolToI1Conversion(typeConverter);
  setupTorchDeviceToLLVMStructConversion(typeConverter);
  setupTorchIntToI64Conversion(typeConverter);
  setupTorchFloatToF64Conversion(typeConverter);
  setupTorchStringToLLVMPtrConversion(typeConverter);
}
