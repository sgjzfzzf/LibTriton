//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchemaLookup.h"
#include "libtriton-core/Conversion/Utils/AOTICAPIDescriptors.h"
#include "libtriton-core/Conversion/Utils/GlobalString.h"
#include "libtriton-core/Conversion/Utils/LibTritonCAPIDescriptors.h"
#include "libtriton-core/Conversion/Utils/StdLibCAPIDescriptors.h"

#include "ATen/core/dispatch/Dispatcher.h"
#include "ATen/core/function_schema.h"
#include "ATen/core/jit_type_base.h"
#include "c10/core/ScalarType.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Rewrite.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

#include <regex>

namespace {

//===----------------------------------------------------------------------===//
// c10 Dispatcher initialization guard
//===----------------------------------------------------------------------===//

/// Ensures c10::Dispatcher has been initialized (primarily to guarantee that
/// the registered operator table is populated before schema lookups).
void ensureC10Registered() {
  static bool registered = false;
  if (!registered) {
    c10::Dispatcher::singleton();
    registered = true;
  }
}

//===----------------------------------------------------------------------===//
// StableIValue building from c10 type info
//===----------------------------------------------------------------------===//

/// Converts an adapted (LLVM-typed) MLIR Value into a type-erased StableIValue
/// (uint64_t) based on the c10 type from the operator schema.
///
/// \param builder  The MLIR op builder.
/// \param c10Type  The c10 type from the op schema argument.
/// \param input    The adapted LLVM-typed MLIR value.
/// \param moduleOp The parent MLIR module.
/// \param loc      The source location.
/// \return The i64 StableIValue on success, or failure.
mlir::FailureOr<mlir::Value> buildStableIValue(mlir::OpBuilder &builder,
                                               const c10::TypePtr &c10Type,
                                               mlir::Value input,
                                               mlir::ModuleOp moduleOp,
                                               mlir::Location loc) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);

  switch (c10Type->kind()) {
  case c10::TypeKind::TensorType: {
    // Tensor: adapted value is a TVMFFIObjectHandle (!llvm.ptr).
    // Convert pointer to i64 directly for the type-erased StableIValue slot.
    // The runtime already uses a handle representation, so PtrToInt is
    // sufficient to pass the handle through the dispatcher slot.
    return mlir::LLVM::PtrToIntOp::create(builder, loc, i64Ty, input)
        .getResult();
  }
  case c10::TypeKind::BoolType: {
    // Bool: zero-extend i1 to i64.
    return mlir::LLVM::ZExtOp::create(builder, loc, i64Ty, input).getResult();
  }
  case c10::TypeKind::IntType:
  case c10::TypeKind::SymIntType: {
    // Int/SymInt: already i64, pass through.
    return input;
  }
  case c10::TypeKind::FloatType:
  case c10::TypeKind::SymFloatType: {
    // Float: bitcast f64 to i64.
    return mlir::LLVM::BitcastOp::create(builder, loc, i64Ty, input)
        .getResult();
  }
  case c10::TypeKind::NoneType: {
    // None: constant 0.
    return mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 0).getResult();
  }
  case c10::TypeKind::OptionalType: {
    // Optional: treat as None → constant 0 for now.
    // TODO: handle non-None optional values based on adapted type.
    return mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 0).getResult();
  }
  case c10::TypeKind::DeviceObjType: {
    // Device: adapted type is llvm.struct<(i32, i32)>.
    // Pack device_type and device_index into a single i64.
    mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::Value devType =
        mlir::LLVM::ExtractValueOp::create(builder, loc, i32Ty, input,
                                           llvm::ArrayRef<int64_t>{0})
            .getResult();
    mlir::Value devIdx =
        mlir::LLVM::ExtractValueOp::create(builder, loc, i32Ty, input,
                                           llvm::ArrayRef<int64_t>{1})
            .getResult();
    mlir::Value devType64 =
        mlir::LLVM::ZExtOp::create(builder, loc, i64Ty, devType);
    mlir::Value devIdx64 =
        mlir::LLVM::ZExtOp::create(builder, loc, i64Ty, devIdx);
    mlir::Value shift = mlir::LLVM::ConstantOp::create(
        builder, loc, i64Ty, mlir::IntegerAttr::get(i64Ty, 32));
    mlir::Value shifted =
        mlir::LLVM::ShlOp::create(builder, loc, devIdx64, shift);
    return mlir::LLVM::OrOp::create(builder, loc, devType64, shifted)
        .getResult();
  }
  case c10::TypeKind::ListType:
  case c10::TypeKind::AnyListType: {
    // List: adapted type is !llvm.ptr (StableListHandle*).
    // Convert pointer to i64.
    return mlir::LLVM::PtrToIntOp::create(builder, loc, i64Ty, input)
        .getResult();
  }
  default:
    return mlir::failure();
  }
}

/// Converts a type-erased StableIValue (uint64_t) back to the appropriate
/// adapted (LLVM-typed) MLIR Value based on the c10 type from the operator
/// schema.
///
/// \param builder  The MLIR op builder.
/// \param c10Type  The c10 type from the op schema result.
/// \param loaded   The i64 StableIValue loaded from the dispatcher stack.
/// \param moduleOp The parent MLIR module.
/// \param loc      The source location.
/// \return The adapted LLVM-typed MLIR value on success, or failure.
mlir::FailureOr<mlir::Value> resolveStableIValue(mlir::OpBuilder &builder,
                                                 const c10::TypePtr &c10Type,
                                                 mlir::Value loaded,
                                                 mlir::ModuleOp moduleOp,
                                                 mlir::Location loc) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);

  switch (c10Type->kind()) {
  case c10::TypeKind::TensorType: {
    // Tensor: i64 → !llvm.ptr (TVMFFIObjectHandle).
    // Reverse of PtrToInt above.
    return mlir::LLVM::IntToPtrOp::create(
               builder, loc, mlir::LLVM::LLVMPointerType::get(ctx), loaded)
        .getResult();
  }
  case c10::TypeKind::BoolType: {
    // Bool: trunc i64 to i1.
    return mlir::LLVM::TruncOp::create(builder, loc,
                                       mlir::IntegerType::get(ctx, 1), loaded)
        .getResult();
  }
  case c10::TypeKind::IntType:
  case c10::TypeKind::SymIntType: {
    // Int/SymInt: already i64, pass through.
    return loaded;
  }
  case c10::TypeKind::FloatType:
  case c10::TypeKind::SymFloatType: {
    // Float: bitcast i64 to f64.
    return mlir::LLVM::BitcastOp::create(builder, loc,
                                         mlir::Float64Type::get(ctx), loaded)
        .getResult();
  }
  case c10::TypeKind::NoneType: {
    // None: no value produced.
    return mlir::Value();
  }
  case c10::TypeKind::OptionalType: {
    // Optional result: pass-through the loaded i64 as-is.
    return loaded;
  }
  case c10::TypeKind::DeviceObjType: {
    // Device: unpack i64 → llvm.struct<(i32, i32)>.
    mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::Value devType =
        mlir::LLVM::TruncOp::create(builder, loc, i32Ty, loaded);
    mlir::Value shift = mlir::LLVM::ConstantOp::create(
        builder, loc, i64Ty, mlir::IntegerAttr::get(i64Ty, 32));
    mlir::Value shifted =
        mlir::LLVM::LShrOp::create(builder, loc, loaded, shift);
    mlir::Value devIdx =
        mlir::LLVM::TruncOp::create(builder, loc, i32Ty, shifted);

    auto structTy = mlir::LLVM::LLVMStructType::getLiteral(ctx, {i32Ty, i32Ty});
    mlir::Value undef = mlir::LLVM::UndefOp::create(builder, loc, structTy);
    mlir::Value s0 =
        mlir::LLVM::InsertValueOp::create(builder, loc, undef, devType,
                                          llvm::ArrayRef<int64_t>{0})
            .getResult();
    return mlir::LLVM::InsertValueOp::create(builder, loc, s0, devIdx,
                                             llvm::ArrayRef<int64_t>{1})
        .getResult();
  }
  case c10::TypeKind::ListType:
  case c10::TypeKind::AnyListType: {
    // List: i64 → !llvm.ptr (StableListHandle*).
    return mlir::LLVM::IntToPtrOp::create(
               builder, loc, mlir::LLVM::LLVMPointerType::get(ctx), loaded)
        .getResult();
  }
  default:
    return mlir::failure();
  }
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Public C API entry point
//===----------------------------------------------------------------------===//

int mLibTritonSchemaDispatchTorchAtenOp(
    MlirOperation op, MlirValue *operands, MlirValue *results,
    MlirConversionPatternRewriter rewriter) {
  // Unwrap the C API types.
  mlir::Operation *mlirOp = unwrap(op);
  mlir::ConversionPatternRewriter &mlirRewriter = *unwrap(rewriter);
  mlir::Location loc = mlirOp->getLoc();
  mlir::MLIRContext *ctx = mlirOp->getContext();

  // Extract the op name.
  llvm::StringRef opName = mlirOp->getName().getStringRef();

  // Match "torch.aten.<OpName>[.<Overload>]" using std::regex.
  static const std::regex re(
      R"(^torch\.aten\.([_a-zA-Z]+)(?:\.([_a-zA-Z]+))?$)");
  std::smatch m;
  std::string opNameCopy = opName.str();
  if (!std::regex_match(opNameCopy, m, re)) {
    return 1;
  }

  const std::string dispatcherOpName = "aten::" + m[1].str();
  const std::string dispatcherOverloadName = m[2].str();

  // Look up the operator schema via c10::Dispatcher.
  c10::OperatorName c10OpName(dispatcherOpName, dispatcherOverloadName);
  auto opHandle = c10::Dispatcher::singleton().findSchema(c10OpName);
  if (!opHandle.has_value()) {
    mlirOp->emitError() << "operator " << dispatcherOpName << "."
                        << dispatcherOverloadName
                        << " which doesn't have a schema registered yet";
    return 1;
  }

  const auto &schema = opHandle->schema();
  const auto &schemaArgs = schema.arguments();
  const auto &schemaReturns = schema.returns();

  // Find the parent module.
  mlir::ModuleOp moduleOp = mlirOp->getParentOfType<mlir::ModuleOp>();
  if (!moduleOp) {
    mlirOp->emitError("op is not inside a module");
    return 1;
  }

  const size_t numInputs = mlirOp->getNumOperands();
  const size_t numResults = mlirOp->getNumResults();
  const size_t maxCount = std::max(numInputs, numResults);

  // Allocate an i64 slot array with maxCount elements on the stack.
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::Value array = mlir::LLVM::AllocaOp::create(
      mlirRewriter, loc, ptrTy, i64Ty,
      mlir::LLVM::ConstantOp::create(mlirRewriter, loc,
                                     mlir::IntegerType::get(ctx, 64), maxCount)
          .getResult());

  // Convert and store each operand into the slot array.
  for (size_t i = 0; i < numInputs; ++i) {
    mlir::Value adaptedVal = unwrap(operands[i]);

    // Determine the c10 type for this argument.
    c10::TypePtr argType =
        (i < schemaArgs.size()) ? schemaArgs[i].type() : nullptr;

    mlir::Value ival;
    if (argType) {
      auto built =
          buildStableIValue(mlirRewriter, argType, adaptedVal, moduleOp, loc);
      if (mlir::failed(built)) {
        mlirOp->emitError("unsupported input type: ")
            << c10::typeKindToString(argType->kind());
        return 1;
      }
      ival = built.value();
    } else {
      // No schema type info: pass the adapted value as-is (assume i64).
      ival = adaptedVal;
    }

    mlir::Value ptr =
        mlir::LLVM::GEPOp::create(mlirRewriter, loc, ptrTy, i64Ty, array, {i});
    mlir::LLVM::StoreOp::create(mlirRewriter, loc, ival, ptr);
  }

  // Get or create the aoti_torch_call_dispatcher function declaration.
  mlir::FailureOr<mlir::LLVM::LLVMFuncOp> calleeOrErr =
      libtriton::conversion::utils::getOrCreateaoti_torch_call_dispatcher(
          moduleOp);
  if (mlir::failed(calleeOrErr)) {
    return 1;
  }

  // Create global string constants for op_name and overload_name.
  mlir::Value opNamePtr = libtriton::conversion::utils::getOrCreateGlobalString(
      mlirRewriter, loc, moduleOp, "op", dispatcherOpName);
  mlir::Value overloadNamePtr =
      libtriton::conversion::utils::getOrCreateGlobalString(
          mlirRewriter, loc, moduleOp, "overload", dispatcherOverloadName);

  // Call aoti_torch_call_dispatcher(opName, overloadName, slotArray).
  mlir::LLVM::CallOp::create(
      mlirRewriter, loc, *calleeOrErr,
      mlir::ValueRange{opNamePtr, overloadNamePtr, array});

  // Load and convert each result from the slot array.
  for (size_t i = 0; i < numResults; ++i) {
    mlir::Value ptr =
        mlir::LLVM::GEPOp::create(mlirRewriter, loc, ptrTy, i64Ty, array, {i});
    mlir::Value loaded =
        mlir::LLVM::LoadOp::create(mlirRewriter, loc, i64Ty, ptr);

    c10::TypePtr retType =
        (i < schemaReturns.size()) ? schemaReturns[i].type() : nullptr;

    mlir::Value result;
    if (retType) {
      auto resolved =
          resolveStableIValue(mlirRewriter, retType, loaded, moduleOp, loc);
      if (mlir::failed(resolved)) {
        mlirOp->emitError("unsupported result type: ")
            << c10::typeKindToString(retType->kind());
        return 1;
      }
      result = resolved.value();
    } else {
      result = loaded;
    }

    results[i] = wrap(result);
  }

  return 0;
}
