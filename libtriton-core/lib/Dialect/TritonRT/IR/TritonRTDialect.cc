#include "libtriton-core/Dialect/TritonRT/IR/TritonRTDialect.h"
#include "libtriton-core/Dialect/TritonRT/IR/TritonRTDialect.cpp.inc"
#include "libtriton-core/Dialect/TritonRT/IR/TritonRTOps.h"
#include "libtriton-core/Dialect/TritonRT/IR/TritonRTTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_OP_CLASSES
#include "libtriton-core/Dialect/TritonRT/IR/TritonRT.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "libtriton-core/Dialect/TritonRT/IR/TritonRTTypes.cpp.inc"

namespace libtriton::triton_rt {

void TritonRTDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "libtriton-core/Dialect/TritonRT/IR/TritonRT.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "libtriton-core/Dialect/TritonRT/IR/TritonRTTypes.cpp.inc"
      >();
}

} // namespace libtriton::triton_rt
