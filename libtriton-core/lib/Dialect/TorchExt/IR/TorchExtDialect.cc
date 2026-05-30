#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtTypes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.cpp.inc"

/// Parses optional async dependencies with an optional leading async keyword:
/// (`async`)? (`[` ssa-id-list `]`)?
static mlir::ParseResult parseAsyncDependencies(
    mlir::OpAsmParser &parser, mlir::Type &asyncTokenType,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand>
        &asyncDependencies) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (mlir::succeeded(parser.parseOptionalKeyword("async"))) {
    if (parser.getNumResults() == 0) {
      return parser.emitError(loc, "needs to be named when marked 'async'");
    }
    asyncTokenType = parser.getBuilder().getType<mlir::gpu::AsyncTokenType>();
  }
  return parser.parseOperandList(asyncDependencies,
                                 mlir::OpAsmParser::Delimiter::OptionalSquare);
}

/// Prints optional async dependencies with an optional leading async keyword.
static void printAsyncDependencies(mlir::OpAsmPrinter &printer,
                                   mlir::Operation * /*op*/,
                                   mlir::Type asyncTokenType,
                                   mlir::OperandRange asyncDependencies) {
  if (asyncTokenType) {
    printer << "async";
  }
  if (!asyncDependencies.empty()) {
    if (asyncTokenType) {
      printer << ' ';
    }
    printer << '[';
    llvm::interleaveComma(asyncDependencies, printer);
    printer << ']';
  }
}

/// Parses optional launch dimension type.
/// (`:` type)?
static mlir::ParseResult parseLaunchDimType(
    mlir::OpAsmParser &parser, mlir::Type &dimTy,
    std::optional<mlir::OpAsmParser::UnresolvedOperand> clusterValue,
    mlir::Type &clusterXTy, mlir::Type &clusterYTy, mlir::Type &clusterZTy) {
  if (mlir::succeeded(parser.parseOptionalColon())) {
    if (parser.parseType(dimTy)) {
      return mlir::failure();
    }
  } else {
    dimTy = mlir::IndexType::get(parser.getContext());
  }

  if (clusterValue.has_value()) {
    clusterXTy = clusterYTy = clusterZTy = dimTy;
  }
  return mlir::success();
}

static void printLaunchDimType(mlir::OpAsmPrinter &printer,
                               mlir::Operation * /*op*/, mlir::Type dimTy,
                               mlir::Value /*clusterValue*/,
                               mlir::Type /*clusterXTy*/,
                               mlir::Type /*clusterYTy*/,
                               mlir::Type /*clusterZTy*/) {
  if (!dimTy.isIndex()) {
    printer << ": " << dimTy;
  }
}

#define GET_OP_CLASSES
#include "libtriton-core/Dialect/TorchExt/IR/TorchExt.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtTypes.cpp.inc"

namespace libtriton::torch_ext {

bool TritonKernelLaunchOp::hasClusterSize() {
  return getClusterSizeX() && getClusterSizeY() && getClusterSizeZ();
}

mlir::LogicalResult TritonKernelLaunchOp::verify() {
  if (!getAsyncDependencies().empty() && !getAsyncToken()) {
    return emitOpError(
        "requires an async token result when async dependencies are present");
  }
  return mlir::success();
}

void TorchExtDialect::initialize() {
  getContext()->loadDialect<mlir::gpu::GPUDialect>();
  addOperations<
#define GET_OP_LIST
#include "libtriton-core/Dialect/TorchExt/IR/TorchExt.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtTypes.cpp.inc"
      >();
}

} // namespace libtriton::torch_ext
