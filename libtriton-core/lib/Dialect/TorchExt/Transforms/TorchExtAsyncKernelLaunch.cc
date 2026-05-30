#include "mlir/IR/MLIRContext.h"
#define GEN_PASS_DEF_ASYNCKERNELLAUNCH
#include "libtriton-core/Dialect/TorchExt/Transforms/TorchExtAsyncKernelLaunch.h"

#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

namespace libtriton::torch_ext {

namespace {

class TritonLaunchAsyncTokenPattern
    : public mlir::OpRewritePattern<TritonKernelLaunchOp> {
public:
  using mlir::OpRewritePattern<TritonKernelLaunchOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(TritonKernelLaunchOp launchOp,
                  mlir::PatternRewriter &rewriter) const final {
    if (!launchOp.getAsyncDependencies().empty() || launchOp.getAsyncToken() ||
        launchOp.getAsyncObject()) {
      return mlir::failure();
    }
    mlir::Type asyncTokenType =
        mlir::gpu::AsyncTokenType::get(rewriter.getContext());
    GetCurrentStreamOp streamOp =
        GetCurrentStreamOp::create(rewriter, launchOp.getLoc(), asyncTokenType,
                                   rewriter.getI8IntegerAttr(-1));
    rewriter.replaceOpWithNewOp<TritonKernelLaunchOp>(
        launchOp, mlir::TypeRange{asyncTokenType},
        llvm::SmallVector<mlir::Value>{streamOp}, launchOp.getKernelAttr(),
        launchOp.getGridSizeX(), launchOp.getGridSizeY(),
        launchOp.getGridSizeZ(), launchOp.getBlockSizeX(),
        launchOp.getBlockSizeY(), launchOp.getBlockSizeZ(),
        launchOp.getClusterSizeX(), launchOp.getClusterSizeY(),
        launchOp.getClusterSizeZ(), launchOp.getDynamicSharedMemorySize(),
        launchOp.getKernelOperands(), mlir::Value{});
    return mlir::success();
  }
};

class AsyncKernelLaunchPass
    : public impl::AsyncKernelLaunchBase<AsyncKernelLaunchPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::ModuleOp moduleOp = getOperation();
    mlir::RewritePatternSet patterns(&context);
    patterns.add<TritonLaunchAsyncTokenPattern>(&context);
    if (mlir::failed(
            mlir::applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace libtriton::torch_ext
