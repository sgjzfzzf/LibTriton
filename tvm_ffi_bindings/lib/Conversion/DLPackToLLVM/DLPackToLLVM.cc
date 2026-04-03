// DLPackToLLVM.cc - Pass that lowers DLPack dialect ops to LLVM dialect.
//
// Currently a scaffold: the conversion target, type converter, and rewrite
// patterns are not yet implemented. The pass is registered via the static
// `kPass` initializer (mlir::PassRegistration), so the explicit
// registerConvertDLPackToLLVMPass() / registerDLPackToLLVMPasses() functions
// are intentional no-ops kept for API consistency with the MLIR pass registry
// convention (callers can call them without knowing whether static registration
// is already in effect).

#include "tvm_ffi_bindings/Conversion/DLPackToLLVM/DLPackToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackDialect.h"
#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackOps.h"

namespace libtriton::dlpack {
namespace {

struct LowerFromMemRefOp
    : public mlir::OpRewritePattern<libtriton::dlpack::FromMemRefOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::dlpack::FromMemRefOp op,
                  mlir::PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
        op, op.getOutput().getType(), op.getInput());
    return mlir::success();
  }
};

struct LowerToMemRefOp
    : public mlir::OpRewritePattern<libtriton::dlpack::ToMemRefOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::dlpack::ToMemRefOp op,
                  mlir::PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
        op, op.getOutput().getType(), op.getInput());
    return mlir::success();
  }
};

class ConvertDLPackToLLVMPass
    : public mlir::PassWrapper<ConvertDLPackToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertDLPackToLLVMPass)

  llvm::StringRef getArgument() const final { return "convert-dlpack-to-llvm"; }

  llvm::StringRef getDescription() const final {
    return "Lower DLPack dialect operations to LLVM dialect";
  }

  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::RewritePatternSet patterns(&context);
    patterns.add<LowerFromMemRefOp, LowerToMemRefOp>(&context);

    mlir::ConversionTarget target(context);
    target.addIllegalDialect<libtriton::dlpack::DLPackDialect>();
    target.addLegalDialect<mlir::BuiltinDialect, mlir::func::FuncDialect>();
    target.markUnknownOpDynamicallyLegal(
        [](mlir::Operation *) { return true; });

    if (failed(mlir::applyPartialConversion(getOperation(), target,
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

static mlir::PassRegistration<ConvertDLPackToLLVMPass> kPass;

} // namespace

std::unique_ptr<mlir::Pass> createConvertDLPackToLLVMPass() {
  return std::make_unique<ConvertDLPackToLLVMPass>();
}

void registerConvertDLPackToLLVMPass() {
  // Registration is handled by static PassRegistration above.
}

void registerDLPackToLLVMPasses() { registerConvertDLPackToLLVMPass(); }

} // namespace libtriton::dlpack
