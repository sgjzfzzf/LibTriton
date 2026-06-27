#include "libtriton-core/Conversion/TorchToLLVM/TorchToLLVM.h"
#include "libtriton-core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch/csrc/stable/c/shim.h"

namespace libtriton::torch {

#define GEN_PASS_DEF_CONVERTTORCHTOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Lower torch.constant.* ops directly to LLVM::ConstantOp
// ---------------------------------------------------------------------------

/// Lowers torch.constant.bool to LLVM::ConstantOp (i1).
class ConvertTorchConstantBoolOp
    : public mlir::OpConversionPattern<mlir::torch::Torch::ConstantBoolOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = mlir::torch::Torch::ConstantBoolOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::ConstantBoolOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type ty = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(
        op, ty, rewriter.getIntegerAttr(ty, op.getValue() ? 1 : 0));
    return mlir::success();
  }
};

/// Lowers torch.constant.int to LLVM::ConstantOp (i64).
class ConvertTorchConstantIntOp
    : public mlir::OpConversionPattern<mlir::torch::Torch::ConstantIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = mlir::torch::Torch::ConstantIntOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::ConstantIntOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type ty = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(
        op, ty, rewriter.getIntegerAttr(ty, op.getValue()));
    return mlir::success();
  }
};

/// Lowers torch.constant.float to LLVM::ConstantOp (f64).
class ConvertTorchConstantFloatOp
    : public mlir::OpConversionPattern<mlir::torch::Torch::ConstantFloatOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = mlir::torch::Torch::ConstantFloatOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::ConstantFloatOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type ty = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(
        op, ty, rewriter.getFloatAttr(ty, op.getValue()));
    return mlir::success();
  }
};

/// Lowers torch.constant.none to LLVM::ConstantOp (i64 0).
class ConvertTorchConstantNoneOp
    : public mlir::OpConversionPattern<mlir::torch::Torch::ConstantNoneOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = mlir::torch::Torch::ConstantNoneOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::ConstantNoneOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type ty = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(
        op, ty, rewriter.getIntegerAttr(ty, 0));
    return mlir::success();
  }
};

/// Lowers torch.constant.device to LLVM struct {i32 device_type, i32
/// device_index}.  The device value *must* be preserved because downstream
/// ops (e.g. torch.aten.empty.memory_format) rely on it to allocate output
/// tensors on the correct device.
class ConvertTorchConstantDeviceOp
    : public mlir::OpConversionPattern<mlir::torch::Torch::ConstantDeviceOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = mlir::torch::Torch::ConstantDeviceOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::ConstantDeviceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Location loc = op.getLoc();
    mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);

    // The result type is {i32 device_type, i32 device_index}.
    mlir::LLVM::LLVMStructType structTy =
        mlir::cast<mlir::LLVM::LLVMStructType>(
            getTypeConverter()->convertType(op.getType()));

    // Parse the device string (e.g. "cuda:0", "cpu") using PyTorch's
    // stable C API.  This correctly handles all device types and returns
    // AOTI-compatible values (matching aoti_torch_device_type_*()).
    llvm::StringRef dev = op.getValue();
    uint32_t deviceType = 0;
    int32_t deviceIndex = 0;
    if (torch_parse_device_string(dev.data(), &deviceType, &deviceIndex)) {
      return op.emitError("failed to parse device string: ") << dev;
    }

    // Build the struct: undef → insert device_type → insert device_index.
    mlir::Value result = mlir::LLVM::UndefOp::create(rewriter, loc, structTy);
    mlir::Value typeVal =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, deviceType);
    result = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, structTy, result, typeVal, llvm::ArrayRef<int64_t>{0});
    mlir::Value idxVal =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, deviceIndex);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        op, structTy, result, idxVal, llvm::ArrayRef<int64_t>{1});
    return mlir::success();
  }
};

// ---------------------------------------------------------------------------
// ConvertTorchToLLVM pass
// ---------------------------------------------------------------------------

class ConvertTorchToLLVMPass
    : public impl::ConvertTorchToLLVMBase<ConvertTorchToLLVMPass> {
public:
  void runOnOperation() final {
    mlir::LLVMTypeConverter typeConverter(&getContext());
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());

    setupBackendTypeConversion(target, typeConverter);

    target.addIllegalOp<
        mlir::torch::Torch::ConstantBoolOp, mlir::torch::Torch::ConstantIntOp,
        mlir::torch::Torch::ConstantFloatOp, mlir::torch::Torch::ConstantNoneOp,
        mlir::torch::Torch::ConstantDeviceOp>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect, mlir::BuiltinDialect>();

    patterns.add<ConvertTorchConstantBoolOp, ConvertTorchConstantIntOp,
                 ConvertTorchConstantFloatOp, ConvertTorchConstantNoneOp,
                 ConvertTorchConstantDeviceOp>(typeConverter,
                                               patterns.getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateTorchToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  setupBackendTypeConversion(target, typeConverter);
  patterns.add<ConvertTorchConstantBoolOp, ConvertTorchConstantIntOp,
               ConvertTorchConstantFloatOp>(typeConverter,
                                            patterns.getContext());
}

} // namespace libtriton::torch
