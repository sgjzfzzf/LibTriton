//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "libtriton-core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "libtriton-core/Conversion/TVMFFIToLLVM/TVMFFICAPIDescriptors.h"
#include "libtriton-core/Conversion/Utils/IConstantUtils.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "libtriton-core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"

namespace libtriton::tvm_ffi {

#define GEN_PASS_DEF_CONVERTTVMFFITOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

namespace {

static mlir::LLVM::LLVMStructType getTVMFFIAnyType(mlir::MLIRContext *context) {
  mlir::Type i32Ty = mlir::IntegerType::get(context, 32);
  mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
  return mlir::LLVM::LLVMStructType::getLiteral(context, {i32Ty, i32Ty, i64Ty},
                                                true);
}

/// Given a void* pointing to a TVMFFIAny ({i32, i32, i64}) struct in an array,
/// return a pointer to its i64 payload field (field index 2).
static mlir::Value getTVMFFIAnyPayloadPtr(mlir::OpBuilder &rewriter,
                                          mlir::Value slotPtr) {
  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(ctx);
  return mlir::LLVM::GEPOp::create(rewriter, slotPtr.getLoc(), ptrTy, anyTy,
                                   slotPtr,
                                   mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
}

//===----------------------------------------------------------------------===//
// Type conversion handlers for packing/unpacking TVM FFI arguments
//===----------------------------------------------------------------------===//

/// CRTP base: auto-generates matches(mlir::Type) from the Torch type parameter.
template <typename TorchType> struct TypeHandlerBase {
  static bool matches(mlir::Type type) { return mlir::isa<TorchType>(type); }
};

/// Helpers for packing/unpacking DLManagedTensor fields.

static mlir::LLVM::LLVMStructType getDLDeviceType(mlir::MLIRContext *ctx) {
  mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
  return mlir::LLVM::LLVMStructType::getLiteral(ctx, {i32Ty, i32Ty}, true);
}

static mlir::LLVM::LLVMStructType getDLDataTypeType(mlir::MLIRContext *ctx) {
  mlir::Type i8Ty = mlir::IntegerType::get(ctx, 8);
  mlir::Type i16Ty = mlir::IntegerType::get(ctx, 16);
  return mlir::LLVM::LLVMStructType::getLiteral(ctx, {i8Ty, i8Ty, i16Ty}, true);
}

static mlir::LLVM::LLVMStructType getDLTensorType(mlir::MLIRContext *ctx) {
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  return mlir::LLVM::LLVMStructType::getLiteral(
      ctx,
      {ptrTy, getDLDeviceType(ctx), mlir::IntegerType::get(ctx, 32),
       getDLDataTypeType(ctx), ptrTy, ptrTy, mlir::IntegerType::get(ctx, 64)},
      true);
}

static mlir::LLVM::LLVMStructType
getDLManagedTensorType(mlir::MLIRContext *ctx) {
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  return mlir::LLVM::LLVMStructType::getLiteral(
      ctx, {getDLTensorType(ctx), ptrTy, ptrTy}, true);
}

static mlir::Value callAOTIGetScalar(mlir::OpBuilder &builder,
                                     mlir::Value tensor,
                                     mlir::LLVM::LLVMFuncOp callee,
                                     mlir::Type resultTy) {
  mlir::Location loc = tensor.getLoc();
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::Type i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::Value slot = mlir::LLVM::AllocaOp::create(
      builder, loc, ptrTy, resultTy,
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));
  mlir::LLVM::CallOp::create(builder, loc, callee,
                             mlir::ValueRange{tensor, slot});
  return mlir::LLVM::LoadOp::create(builder, loc, resultTy, slot).getResult();
}

struct BaseTensorHandler : TypeHandlerBase<mlir::torch::Torch::BaseTensorType> {
  /// Pack an AtenTensorHandle (input) into a TVMFFIAny slot (ptr).
  ///
  /// Flow:
  /// 1. Extract tensor properties via aoti_torch_get_* functions.
  /// 2. Reverse-map Torch dtype/device → DLPack dtype/device.
  /// 3. Heap-allocate a DLManagedTensor, fill with extracted properties.
  /// 4. Convert to TVMFFIObjectHandle via TVMFFITensorFromDLPack.
  /// 5. Store type_index=kTVMFFITensor(70) and payload=handle in TVMFFIAny.
  static mlir::LogicalResult store(mlir::OpBuilder &builder, mlir::Value input,
                                   mlir::Value ptr) {
    using namespace libtriton::tvm_ffi::capi;
    mlir::Location loc = input.getLoc();
    mlir::MLIRContext *context = builder.getContext();

    mlir::Type i8Ty = mlir::IntegerType::get(context, 8);
    mlir::Type i16Ty = mlir::IntegerType::get(context, 16);
    mlir::Type i32Ty = mlir::IntegerType::get(context, 32);
    mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
    mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);

    mlir::ModuleOp moduleOp =
        input.getDefiningOp()->getParentOfType<mlir::ModuleOp>();

    // Step 1: Declare all CAPI functions.
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getDataPtrFn =
        getOrCreateAOTITorchGetDataPtr(moduleOp);
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getDimFn =
        getOrCreateAOTITorchGetDim(moduleOp);
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getSizesFn =
        getOrCreateAOTITorchGetSizes(moduleOp);
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getStridesFn =
        getOrCreateAOTITorchGetStrides(moduleOp);
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getDtypeFn =
        getOrCreateAOTITorchGetDtype(moduleOp);
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getDevTypeFn =
        getOrCreateAOTITorchGetDeviceType(moduleOp);
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getDevIdxFn =
        getOrCreateAOTITorchGetDeviceIndex(moduleOp);
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getByteOffFn =
        getOrCreateAOTITorchGetStorageOffset(moduleOp);
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> tvmFromDlpackFn =
        getOrCreateTVMFFITensorFromDLPack(moduleOp);
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> dlpackDeleterFn =
        getOrCreateDLManagedTensorDeleter(moduleOp);

    if (mlir::failed(getDataPtrFn) || mlir::failed(getDimFn) ||
        mlir::failed(getSizesFn) || mlir::failed(getStridesFn) ||
        mlir::failed(getDtypeFn) || mlir::failed(getDevTypeFn) ||
        mlir::failed(getDevIdxFn) || mlir::failed(getByteOffFn) ||
        mlir::failed(tvmFromDlpackFn) || mlir::failed(dlpackDeleterFn)) {
      return mlir::failure();
    }

    // Step 2: Extract tensor properties via aoti_torch_get_*.
    mlir::Value ndim_i64 = callAOTIGetScalar(builder, input, *getDimFn, i64Ty);
    mlir::Value sizesPtr =
        callAOTIGetScalar(builder, input, *getSizesFn, ptrTy);
    mlir::Value stridesPtr =
        callAOTIGetScalar(builder, input, *getStridesFn, ptrTy);
    mlir::Value dataPtr =
        callAOTIGetScalar(builder, input, *getDataPtrFn, ptrTy);
    mlir::Value torchDtype =
        callAOTIGetScalar(builder, input, *getDtypeFn, i32Ty);
    mlir::Value torchDevType =
        callAOTIGetScalar(builder, input, *getDevTypeFn, i32Ty);
    mlir::Value devIdx = callAOTIGetScalar(builder, input, *getDevIdxFn, i32Ty);
    mlir::Value byteOff =
        callAOTIGetScalar(builder, input, *getByteOffFn, i64Ty);

    // Step 3: Reverse-map Torch dtype/device → DLPack.
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> reverseDtypeFn =
        getOrCreateTorchToTVMFFIDtype(moduleOp);
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> reverseDeviceFn =
        getOrCreateTorchToTVMFFIDevice(moduleOp);
    if (mlir::failed(reverseDtypeFn) || mlir::failed(reverseDeviceFn)) {
      return mlir::failure();
    }
    mlir::Value dlDataTypeVal =
        mlir::LLVM::CallOp::create(builder, loc, *reverseDtypeFn,
                                   mlir::ValueRange{torchDtype})
            .getResult();
    mlir::Value dlpackCode =
        mlir::LLVM::ExtractValueOp::create(builder, loc, dlDataTypeVal, 0);
    mlir::Value dlpackBits =
        mlir::LLVM::ExtractValueOp::create(builder, loc, dlDataTypeVal, 1);
    mlir::Value dlpackLanes =
        mlir::LLVM::ExtractValueOp::create(builder, loc, dlDataTypeVal, 2);
    mlir::Value dlpackDevice =
        mlir::LLVM::CallOp::create(builder, loc, *reverseDeviceFn,
                                   mlir::ValueRange{torchDevType})
            .getResult();

    // Step 4: DLManagedTensor LLVM struct type.
    mlir::LLVM::LLVMStructType dlManagedTy = getDLManagedTensorType(context);

    // Step 5: malloc(sizeof(DLManagedTensor)) and fill fields.
    mlir::Value mallocSize = conversion::utils::emitI64Constant(
        builder, loc, context, sizeof(DLManagedTensor));
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> mallocFn =
        getOrCreateMalloc(moduleOp);
    if (mlir::failed(mallocFn)) {
      return mlir::failure();
    }
    mlir::Value dlManPtr =
        mlir::LLVM::CallOp::create(builder, loc, *mallocFn,
                                   mlir::ValueRange{mallocSize})
            .getResult();

    mlir::Value ndim_i32 =
        mlir::LLVM::TruncOp::create(builder, loc, i32Ty, ndim_i64).getResult();

    auto storeDlField = [&](mlir::Value val,
                            llvm::ArrayRef<mlir::LLVM::GEPArg> gepIndices) {
      mlir::Value fieldPtr = mlir::LLVM::GEPOp::create(
          builder, dlManPtr.getLoc(), ptrTy, dlManagedTy, dlManPtr, gepIndices);
      mlir::LLVM::StoreOp::create(builder, loc, val, fieldPtr);
    };

    storeDlField(dataPtr, {0, 0, 0});         // dl_tensor.data
    storeDlField(dlpackDevice, {0, 0, 1, 0}); // dl_tensor.device.type
    storeDlField(devIdx, {0, 0, 1, 1});       // dl_tensor.device.id
    storeDlField(ndim_i32, {0, 0, 2});        // dl_tensor.ndim
    storeDlField(dlpackCode, {0, 0, 3, 0});   // dl_tensor.dtype.code
    storeDlField(dlpackBits, {0, 0, 3, 1});   // dl_tensor.dtype.bits
    storeDlField(dlpackLanes, {0, 0, 3, 2});  // dl_tensor.dtype.lanes
    storeDlField(sizesPtr, {0, 0, 4});        // dl_tensor.shape
    storeDlField(stridesPtr, {0, 0, 5});      // dl_tensor.strides
    storeDlField(byteOff, {0, 0, 6});         // dl_tensor.byte_offset

    // manager_ctx = AtenTensorHandle.
    mlir::Value ctxFieldPtr =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlManagedTy, dlManPtr,
                                  mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 1});
    mlir::LLVM::StoreOp::create(builder, loc, input, ctxFieldPtr);

    // deleter = @mLibTritonDLManagedTensorDeleter.
    mlir::Value deleterVal = mlir::LLVM::AddressOfOp::create(
        builder, loc, ptrTy, dlpackDeleterFn->getSymName());
    mlir::Value deleterFieldPtr =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlManagedTy, dlManPtr,
                                  mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
    mlir::LLVM::StoreOp::create(builder, loc, deleterVal, deleterFieldPtr);

    // Step 6: Call TVMFFITensorFromDLPack.
    mlir::Value objHandleSlot = mlir::LLVM::AllocaOp::create(
        builder, loc, ptrTy, ptrTy,
        conversion::utils::emitI64Constant(builder, loc, context, 1));
    mlir::LLVM::CallOp::create(
        builder, loc, *tvmFromDlpackFn,
        mlir::ValueRange{dlManPtr,
                         mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, 0),
                         mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, 0),
                         objHandleSlot});
    mlir::Value objHandle =
        mlir::LLVM::LoadOp::create(builder, loc, ptrTy, objHandleSlot);

    // Step 7: Store into TVMFFIAny: type_index=kTVMFFITensor(70),
    // payload=handle.
    mlir::Value typeIdxPtr = mlir::LLVM::GEPOp::create(
        builder, loc, ptrTy, getTVMFFIAnyType(context), ptr,
        mlir::ArrayRef<mlir::LLVM::GEPArg>{0});
    mlir::LLVM::StoreOp::create(
        builder, loc,
        mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, kTVMFFITensor),
        typeIdxPtr);

    mlir::Value payloadPtr = getTVMFFIAnyPayloadPtr(builder, ptr);
    mlir::Value payloadI64 =
        mlir::LLVM::PtrToIntOp::create(builder, loc, i64Ty, objHandle);
    mlir::LLVM::StoreOp::create(builder, loc, payloadI64, payloadPtr);

    return mlir::success();
  }
  static mlir::FailureOr<mlir::Value> load(mlir::OpBuilder &builder,
                                           mlir::Value ptr) {
    mlir::Location loc = ptr.getLoc();
    mlir::MLIRContext *context = builder.getContext();
    mlir::Type i8Ty = mlir::IntegerType::get(context, 8);
    mlir::Type i32Ty = mlir::IntegerType::get(context, 32);
    mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
    mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);

    //===------------------------------------------------------------------===//
    // Define DLTensor LLVM struct type  (packed, matching C layout).
    //===------------------------------------------------------------------===//
    mlir::LLVM::LLVMStructType dltensorTy = getDLTensorType(context);

    // Load DLTensor* from the TVMFFIAny payload.
    // The payload may be a direct DLTensor* (kTVMFFIDLTensorPtr=7) or a
    // TVMFFIObject* (type_index >= 64). For objects, the DLTensor starts
    // at offset sizeof(TVMFFIObject)=24 from the object ptr.
    mlir::Value typeIndexPtr = mlir::LLVM::GEPOp::create(
        builder, loc, ptrTy, getTVMFFIAnyType(context), ptr,
        mlir::ArrayRef<mlir::LLVM::GEPArg>{0});
    mlir::Value typeIndex =
        mlir::LLVM::LoadOp::create(builder, loc, i32Ty, typeIndexPtr);
    mlir::Value payloadPtr = getTVMFFIAnyPayloadPtr(builder, ptr);
    mlir::Value rawPayload =
        mlir::LLVM::LoadOp::create(builder, loc, i64Ty, payloadPtr);
    mlir::Value rawPtr =
        mlir::LLVM::IntToPtrOp::create(builder, loc, ptrTy, rawPayload);

    mlir::Value tensorTypeIndex =
        mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, 70);
    mlir::Value isTensorObj =
        mlir::LLVM::ICmpOp::create(builder, loc, mlir::LLVM::ICmpPredicate::eq,
                                   typeIndex, tensorTypeIndex);
    mlir::Value rawAsI64 =
        mlir::LLVM::PtrToIntOp::create(builder, loc, i64Ty, rawPtr);
    mlir::Value objOffset = mlir::LLVM::ConstantOp::create(
        builder, loc, i64Ty, sizeof(TVMFFIObject));
    mlir::Value offsetI64 =
        mlir::LLVM::AddOp::create(builder, loc, rawAsI64, objOffset);
    mlir::Value offsetPtr =
        mlir::LLVM::IntToPtrOp::create(builder, loc, ptrTy, offsetI64);
    mlir::Value dltensorRawPtr = mlir::LLVM::SelectOp::create(
        builder, loc, isTensorObj, offsetPtr, rawPtr);

    // Load DLTensor fields via GEP + Load.
    auto field = [&](mlir::Type ty,
                     llvm::ArrayRef<mlir::LLVM::GEPArg> indices) {
      mlir::Value fieldPtr = mlir::LLVM::GEPOp::create(
          builder, loc, ptrTy, dltensorTy, dltensorRawPtr, indices);
      return mlir::LLVM::LoadOp::create(builder, loc, ty, fieldPtr);
    };
    mlir::Value data = field(ptrTy, {0, 0});
    mlir::Value dlDevType = field(i32Ty, {0, 1, 0});
    mlir::Value deviceId = field(i32Ty, {0, 1, 1});
    mlir::Value ndim = field(i32Ty, {0, 2});
    mlir::Value dtypeCode = field(i8Ty, {0, 3, 0});
    mlir::Value dtypeBits = field(i8Ty, {0, 3, 1});
    mlir::Value shape = field(ptrTy, {0, 4});
    mlir::Value strides = field(ptrTy, {0, 5});
    mlir::Value byteOff = field(i64Ty, {0, 6});

    //===------------------------------------------------------------------===//
    // Map DLPack dtype → Torch dtype via runtime helper.
    //===------------------------------------------------------------------===//
    mlir::ModuleOp moduleOp =
        ptr.getDefiningOp()->getParentOfType<mlir::ModuleOp>();
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> dtypeHelper =
        capi::getOrCreateTVMFFIToTorchType(moduleOp);
    if (mlir::failed(dtypeHelper)) {
      return mlir::failure();
    }
    mlir::Value torchDtype =
        mlir::LLVM::CallOp::create(builder, loc, *dtypeHelper,
                                   mlir::ValueRange{dtypeCode, dtypeBits})
            .getResult();

    //===------------------------------------------------------------------===//
    // Map DLPack device type → Torch device type via runtime helper.
    //===------------------------------------------------------------------===//
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> devTypeFn =
        capi::getOrCreateTVMFFIDeviceToTorchDeviceType(moduleOp);
    if (mlir::failed(devTypeFn)) {
      return mlir::failure();
    }
    mlir::Value deviceType =
        mlir::LLVM::CallOp::create(builder, loc, *devTypeFn,
                                   mlir::ValueRange{dlDevType})
            .getResult();

    //===------------------------------------------------------------------===//
    // Call aoti_torch_create_tensor_from_blob with all extracted fields.
    //===------------------------------------------------------------------===//
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> createFn =
        capi::getOrCreateAOTITorchCreateTensorFromBlob(moduleOp);
    if (mlir::failed(createFn)) {
      return mlir::failure();
    }
    mlir::Value handleSlot = mlir::LLVM::AllocaOp::create(
        builder, loc, ptrTy, ptrTy,
        conversion::utils::emitI64Constant(builder, loc, context, 1));
    mlir::Value ndimI64 = mlir::LLVM::SExtOp::create(builder, loc, i64Ty, ndim);

    mlir::LLVM::CallOp::create(builder, loc, *createFn,
                               mlir::ValueRange{data, ndimI64, shape, strides,
                                                byteOff, torchDtype, deviceType,
                                                deviceId, handleSlot});

    return mlir::LLVM::LoadOp::create(builder, loc, ptrTy, handleSlot)
        .getResult();
  }
};

struct BoolHandler : TypeHandlerBase<mlir::torch::Torch::BoolType> {
  static mlir::LogicalResult store(mlir::OpBuilder &builder, mlir::Value input,
                                   mlir::Value ptr) {
    mlir::Location loc = input.getLoc();
    mlir::Value payloadPtr = getTVMFFIAnyPayloadPtr(builder, ptr);
    mlir::Value ext = mlir::LLVM::ZExtOp::create(
        builder, loc, mlir::IntegerType::get(builder.getContext(), 64), input);
    mlir::LLVM::StoreOp::create(builder, loc, ext, payloadPtr);
    return mlir::success();
  }
  static mlir::FailureOr<mlir::Value> load(mlir::OpBuilder &builder,
                                           mlir::Value ptr) {
    mlir::Location loc = ptr.getLoc();
    mlir::Value payload = getTVMFFIAnyPayloadPtr(builder, ptr);
    mlir::Value loaded = mlir::LLVM::LoadOp::create(
        builder, loc, mlir::IntegerType::get(builder.getContext(), 64),
        payload);
    return mlir::LLVM::TruncOp::create(
               builder, loaded.getLoc(),
               mlir::IntegerType::get(builder.getContext(), 1), loaded)
        .getResult();
  }
};

struct IntHandler : TypeHandlerBase<mlir::torch::Torch::IntType> {
  static mlir::LogicalResult store(mlir::OpBuilder &builder, mlir::Value input,
                                   mlir::Value ptr) {
    mlir::Value payload = getTVMFFIAnyPayloadPtr(builder, ptr);
    mlir::LLVM::StoreOp::create(builder, input.getLoc(), input, payload);
    return mlir::success();
  }
  static mlir::FailureOr<mlir::Value> load(mlir::OpBuilder &builder,
                                           mlir::Value ptr) {
    mlir::Value payload = getTVMFFIAnyPayloadPtr(builder, ptr);
    return mlir::LLVM::LoadOp::create(
               builder, ptr.getLoc(),
               mlir::IntegerType::get(builder.getContext(), 64), payload)
        .getResult();
  }
};

struct FloatHandler : TypeHandlerBase<mlir::torch::Torch::FloatType> {
  static mlir::LogicalResult store(mlir::OpBuilder &builder, mlir::Value input,
                                   mlir::Value ptr) {
    mlir::Location loc = input.getLoc();
    mlir::Value payload = getTVMFFIAnyPayloadPtr(builder, ptr);
    mlir::Value bits = mlir::LLVM::BitcastOp::create(
        builder, loc, mlir::IntegerType::get(builder.getContext(), 64), input);
    mlir::LLVM::StoreOp::create(builder, loc, bits, payload);
    return mlir::success();
  }
  static mlir::FailureOr<mlir::Value> load(mlir::OpBuilder &builder,
                                           mlir::Value ptr) {
    mlir::Location loc = ptr.getLoc();
    mlir::Value payload = getTVMFFIAnyPayloadPtr(builder, ptr);
    mlir::Value loaded = mlir::LLVM::LoadOp::create(
        builder, loc, mlir::IntegerType::get(builder.getContext(), 64),
        payload);
    return mlir::LLVM::BitcastOp::create(
               builder, loc, mlir::Float64Type::get(builder.getContext()),
               loaded)
        .getResult();
  }
};

struct NoneHandler : TypeHandlerBase<mlir::torch::Torch::NoneType> {
  static mlir::LogicalResult store(mlir::OpBuilder &builder, mlir::Value,
                                   mlir::Value ptr) {
    mlir::Location loc = ptr.getLoc();
    mlir::Value payload = getTVMFFIAnyPayloadPtr(builder, ptr);
    mlir::LLVM::StoreOp::create(builder, loc,
                                conversion::utils::emitI64Constant(
                                    builder, loc, builder.getContext(), 0),
                                payload);
    return mlir::success();
  }
  static mlir::FailureOr<mlir::Value> load(mlir::OpBuilder &, mlir::Value) {
    // None → no value produced.
    return mlir::Value();
  }
};

//===----------------------------------------------------------------------===//
// Variadic dispatch: folds over handlers, short-circuits on first match
//===----------------------------------------------------------------------===//

template <typename... Handlers> struct TypeDispatch {
  static mlir::LogicalResult store(mlir::Type type, mlir::OpBuilder &builder,
                                   mlir::Value input, mlir::Value ptr) {
    mlir::LogicalResult result = mlir::failure();
    auto tryHandler = [&](auto handler) {
      if (mlir::failed(result) && decltype(handler)::matches(type)) {
        result = decltype(handler)::store(builder, input, ptr);
      }
    };
    (tryHandler(Handlers{}), ...);
    return result;
  }

  static mlir::FailureOr<mlir::Value>
  load(mlir::Type type, mlir::OpBuilder &builder, mlir::Value ptr) {
    mlir::FailureOr<mlir::Value> result = mlir::failure();
    auto tryHandler = [&](auto handler) {
      if (mlir::failed(result) && decltype(handler)::matches(type)) {
        result = decltype(handler)::load(builder, ptr);
      }
    };
    (tryHandler(Handlers{}), ...);
    return result;
  }
};

using AllHandlers = TypeDispatch<BaseTensorHandler, BoolHandler, IntHandler,
                                 FloatHandler, NoneHandler>;

/// Converts tvm_ffi.func to func.func by converting the function type
/// signature through the type converter and inlining the region body.
class ConvertFuncOp : public mlir::OpConversionPattern<FuncOp> {
public:
  ConvertFuncOp(mlir::LLVMTypeConverter &typeConverter,
                mlir::MLIRContext *context)
      : mlir::OpConversionPattern<FuncOp>(typeConverter, context) {}

  mlir::LogicalResult
  matchAndRewrite(FuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = rewriter.getContext();

    // Without support for tvm_ffi.Array, at most one return value is allowed.
    assert(op.getFunctionType().getResults().size() <= 1 &&
           "Without the support of `tvm.ffi.Array`, we only support at most "
           "one return value temporarily.");

    // TVM-FFI C ABI: int32_t(void*, void*, int32_t, void*)
    mlir::IntegerType i32Ty = rewriter.getIntegerType(32);
    mlir::LLVM::LLVMPointerType ptrTy =
        mlir::LLVM::LLVMPointerType::get(context);
    mlir::LLVM::LLVMFunctionType funcType =
        mlir::LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, ptrTy, i32Ty, ptrTy});

    // Create the LLVM function, then build all blocks from scratch.
    std::string tvmffiFuncName =
        llvm::formatv("__tvm_ffi_{0}", op.getSymName());
    mlir::LLVM::LLVMFuncOp llvmFunc = mlir::LLVM::LLVMFuncOp::create(
        rewriter, loc, tvmffiFuncName, funcType, mlir::LLVM::Linkage::External);
    mlir::Region &region = llvmFunc.getBody();
    mlir::IRMapping mapping;

    // Create all blocks first: one per original block.
    for (mlir::Block &blk : op.getBody()) {
      mlir::Block *newBlock = rewriter.createBlock(&region);
      mapping.map(&blk, newBlock);
    }

    // The first block is the entry; add ABI arguments to it.
    // TVM-FFI C ABI: int32_t(void* handle, TVMFFIAny* args, int32_t num_args,
    // TVMFFIAny* result)
    mlir::Block *entryBlock = &region.front();
    entryBlock->addArgument(ptrTy, loc); // handle (unused)
    mlir::Value argsPtr = entryBlock->addArgument(ptrTy, loc); // args
    entryBlock->addArgument(i32Ty, loc); // num_args (unused)
    mlir::Value retPtr = entryBlock->addArgument(ptrTy, loc); // result

    // Load input payloads via AllHandlers::load, cast to Torch, map to
    // original block args.
    rewriter.setInsertionPointToStart(entryBlock);
    mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(context);
    mlir::Block &entry = op.getBody().front();
    for (auto [i, arg] : llvm::enumerate(entry.getArguments())) {
      mlir::Value slot = mlir::LLVM::GEPOp::create(
          rewriter, loc, ptrTy, anyTy, argsPtr,
          mlir::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)});
      mlir::Type argTy = arg.getType();
      mlir::FailureOr<mlir::Value> loaded =
          AllHandlers::load(argTy, rewriter, slot);
      if (mlir::failed(loaded)) {
        return op.emitError("unsupported input type: ") << argTy;
      }
      mlir::Value casted = mlir::UnrealizedConversionCastOp::create(
                               rewriter, loc, argTy, *loaded)
                               .getResult(0);
      mapping.map(arg, casted);
    }

    // Clone every op from every block into its corresponding new block.

    for (mlir::Block &blk : op.getBody()) {
      mlir::Block *dest = mapping.lookupOrDefault(&blk);
      rewriter.setInsertionPointToEnd(dest);
      for (mlir::Operation &operation : llvm::make_early_inc_range(blk)) {
        if (ReturnOp returnOp = mlir::dyn_cast<ReturnOp>(&operation)) {
          for (mlir::Value operand : returnOp.getOperands()) {
            mlir::Value retVal = mapping.lookupOrDefault(operand);
            mlir::Type operandTy = operand.getType();
            mlir::Value casted =
                mlir::UnrealizedConversionCastOp::create(
                    rewriter, loc, getTypeConverter()->convertType(operandTy),
                    retVal)
                    .getResult(0);
            if (mlir::failed(
                    AllHandlers::store(operandTy, rewriter, casted, retPtr))) {
              return op.emitError("unsupported return type: ") << operandTy;
            }
          }
          mlir::LLVM::ConstantOp cnst =
              mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, 0);
          mlir::LLVM::ReturnOp::create(rewriter, loc, cnst.getResult());
        } else {
          rewriter.clone(operation, mapping);
        }
      }
    }

    rewriter.replaceOp(op, llvmFunc);
    return mlir::success();
  }
};

class ConvertTVMFFIToLLVMPass
    : public impl::ConvertTVMFFIToLLVMBase<ConvertTVMFFIToLLVMPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::ConversionTarget target(context);
    mlir::LLVMTypeConverter typeConverter(&context);
    mlir::RewritePatternSet patterns(&context);

    libtriton::torch::setupBackendTypeConversion(target, typeConverter);
    target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp>();
    populateTVMFFIToLLVMConversionPatterns(target, typeConverter, patterns);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct TVMFFIToLLVMDialectInterface
    : public mlir::ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(
      mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
      mlir::RewritePatternSet &patterns) const final {
    libtriton::torch::setupBackendTypeConversion(target, typeConverter);
    populateTVMFFIToLLVMConversionPatterns(target, typeConverter, patterns);
  }
};

} // namespace

void populateTVMFFIToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertFuncOp>(typeConverter, patterns.getContext());
  target.addIllegalDialect<TVMFFIDialect>();
  target.addLegalDialect<mlir::BuiltinDialect, mlir::func::FuncDialect,
                         mlir::LLVM::LLVMDialect,
                         mlir::torch::Torch::TorchDialect>();
}

void registerConvertTVMFFIToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, libtriton::tvm_ffi::TVMFFIDialect *dialect) {
        dialect->addInterfaces<TVMFFIToLLVMDialectInterface>();
      });
}

} // namespace libtriton::tvm_ffi
