#include "libtriton-core/Conversion/TorchExtToLLVM/TorchExtToLLVM.h"
#include "SchemaLookup.h"
#include "libtriton-core/Conversion/Utils/LibTritonCAPIDescriptors.h"
#include "libtriton-core/Conversion/Utils/StableCAPIDescriptors.h"
#include "libtriton-core/Conversion/Utils/StdLibCAPIDescriptors.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "libtriton-core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Rewrite.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

namespace libtriton::torchext {

#define GEN_PASS_DEF_CONVERTTORCHEXTTOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

namespace {

/// Converts torch.aten.* ops directly to aoti_torch_call_dispatcher() LLVM
/// calls, preserving the original Torch dialect op schema for downstream
/// inspection. The op name is regex-matched to extract the dispatcher
/// op_name (e.g. "aten::empty_like") and optional overload_name.
class ConvertAtenDispatcherOp : public mlir::ConversionPattern {
public:
  ConvertAtenDispatcherOp(const mlir::TypeConverter &typeConverter,
                          mlir::MLIRContext *context)
      : mlir::ConversionPattern(typeConverter,
                                mlir::Pattern::MatchAnyOpTypeTag(),
                                /*benefit=*/1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Wrap adapted operands for the C API.
    llvm::SmallVector<MlirValue> wrappedOperands;
    for (auto v : operands) {
      wrappedOperands.push_back(wrap(v));
    }

    // Allocate output array (handles zero results gracefully).
    uint32_t numResults = op->getNumResults();
    llvm::SmallVector<MlirValue> mlirResults(numResults, {nullptr});

    // Delegate the full lowering to SchemaLookup.
    if (mLibTritonSchemaDispatchTorchAtenOp(wrap(op), wrappedOperands.data(),
                                            mlirResults.data(),
                                            wrap(&rewriter)) != 0) {
      return mlir::failure();
    }

    // Unwrap results and hand them back to the dialect conversion framework.
    llvm::SmallVector<mlir::Value> results;
    for (auto mv : mlirResults) {
      results.push_back(unwrap(mv));
    }
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// StableList operations
//===----------------------------------------------------------------------===//

/// Converts torch.prim.ListConstruct to torch_new_list_reserve_size() +
/// torch_list_push_back() LLVM calls.
class ConvertPrimListConstructOp
    : public mlir::OpConversionPattern<
          mlir::torch::Torch::PrimListConstructOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::PrimListConstructOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = op.getContext();
    mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
    mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);

    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return op.emitError("op is not inside a module");
    }

    int64_t numElements = op.getElements().size();

    // Allocate a stack slot for the StableListHandle* output.
    mlir::Value listPtr = mlir::LLVM::AllocaOp::create(
        rewriter, loc, ptrTy, ptrTy,
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1));

    // Call torch_new_list_reserve_size(numElements, &listHandle).
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> newListCallee =
        libtriton::conversion::utils::getOrCreateTorchNewListReserveSize(
            moduleOp);
    if (mlir::failed(newListCallee)) {
      return mlir::failure();
    }
    mlir::Value numElementsVal =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, numElements);
    mlir::LLVM::CallOp::create(rewriter, loc, *newListCallee,
                               mlir::ValueRange{numElementsVal, listPtr});

    // Load the newly created list handle.
    mlir::Value listHandle =
        mlir::LLVM::LoadOp::create(rewriter, loc, ptrTy, listPtr);

    // Get or create torch_list_push_back.
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> pushBackCallee =
        libtriton::conversion::utils::getOrCreateTorchListPushBack(moduleOp);
    if (mlir::failed(pushBackCallee)) {
      return mlir::failure();
    }

    // Push each element into the list as a type-erased StableIValue.
    // Since this is not a dispatcher call, convert by checking the adapted
    // LLVM type directly rather than via a schema.
    for (auto pair : llvm::zip(op.getElements(), adaptor.getElements())) {
      mlir::Value origElem = std::get<0>(pair);
      mlir::Value adaptedElem = std::get<1>(pair);

      mlir::Type adaptedType = adaptedElem.getType();
      mlir::Value ival;
      if (adaptedType.isInteger(64)) {
        ival = adaptedElem;
      } else if (adaptedType.isInteger(1)) {
        ival = mlir::LLVM::ZExtOp::create(rewriter, loc, i64Ty, adaptedElem)
                   .getResult();
      } else if (adaptedType.isF64()) {
        ival = mlir::LLVM::BitcastOp::create(rewriter, loc, i64Ty, adaptedElem)
                   .getResult();
      } else if (mlir::isa<mlir::LLVM::LLVMPointerType>(adaptedType)) {
        ival = mlir::LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, adaptedElem)
                   .getResult();
      } else if (mlir::isa<mlir::LLVM::LLVMStructType>(adaptedType)) {
        // Device type: pack {i32, i32} struct into a single i64.
        mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
        mlir::Value devType =
            mlir::LLVM::ExtractValueOp::create(
                rewriter, loc, i32Ty, adaptedElem, llvm::ArrayRef<int64_t>{0})
                .getResult();
        mlir::Value devIdx =
            mlir::LLVM::ExtractValueOp::create(
                rewriter, loc, i32Ty, adaptedElem, llvm::ArrayRef<int64_t>{1})
                .getResult();
        mlir::Value devType64 =
            mlir::LLVM::ZExtOp::create(rewriter, loc, i64Ty, devType);
        mlir::Value devIdx64 =
            mlir::LLVM::ZExtOp::create(rewriter, loc, i64Ty, devIdx);
        mlir::Value shift = mlir::LLVM::ConstantOp::create(
            rewriter, loc, i64Ty, mlir::IntegerAttr::get(i64Ty, 32));
        mlir::Value shifted =
            mlir::LLVM::ShlOp::create(rewriter, loc, devIdx64, shift);
        ival = mlir::LLVM::OrOp::create(rewriter, loc, devType64, shifted)
                   .getResult();
      } else {
        return op.emitError("unsupported element type");
      }

      mlir::LLVM::CallOp::create(rewriter, loc, *pushBackCallee,
                                 mlir::ValueRange{listHandle, ival});
    }

    rewriter.replaceOp(op, listHandle);
    return mlir::success();
  }
};

/// Converts torchext.aoti.ListDeleteList to torch_delete_list() LLVM call.
class ConvertListDeleteListOp
    : public mlir::OpConversionPattern<ListDeleteListOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ListDeleteListOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = op.getContext();

    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return op.emitError("op is not inside a module");
    }

    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> calleeOrErr =
        libtriton::conversion::utils::getOrCreateTorchDeleteList(moduleOp);
    if (mlir::failed(calleeOrErr)) {
      return mlir::failure();
    }

    mlir::LLVM::CallOp::create(rewriter, loc, *calleeOrErr,
                               mlir::ValueRange{adaptor.getList()});
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ConvertTorchExtToLLVMPass
    : public impl::ConvertTorchExtToLLVMBase<ConvertTorchExtToLLVMPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::ConversionTarget target(context);
    mlir::LLVMTypeConverter typeConverter(&context);
    mlir::RewritePatternSet patterns(&context);
    torch::setupBackendTypeConversion(target, typeConverter);
    populateTorchExtToLLVMConversionPatterns(target, typeConverter, patterns);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct TorchExtToLLVMDialectInterface
    : public mlir::ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(
      mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
      mlir::RewritePatternSet &patterns) const final {
    // Setup type conversion for torch types before adding patterns, so that
    // the type converter can handle Torch tensor/bool/int/float/optional etc.
    // types when patterns query adaptor types.
    torch::setupBackendTypeConversion(target, typeConverter);
    populateTorchExtToLLVMConversionPatterns(target, typeConverter, patterns);
  }
};

} // namespace

void populateTorchExtToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertAtenDispatcherOp, ConvertPrimListConstructOp,
               ConvertListDeleteListOp>(typeConverter, patterns.getContext());
  target.addIllegalOp<mlir::torch::Torch::PrimListConstructOp>();
  target.addIllegalDialect<TorchExtDialect>();
  target.addLegalDialect<mlir::BuiltinDialect, mlir::LLVM::LLVMDialect>();
}

void registerConvertTorchExtToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx,
                            libtriton::torchext::TorchExtDialect *dialect) {
    dialect->addInterfaces<TorchExtToLLVMDialectInterface>();
  });
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, mlir::torch::Torch::TorchDialect *dialect) {
        dialect->addInterfaces<TorchExtToLLVMDialectInterface>();
      });
}

} // namespace libtriton::torchext
