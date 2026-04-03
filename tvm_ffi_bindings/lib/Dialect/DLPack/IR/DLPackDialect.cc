#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackDialect.cpp.inc"
#include "llvm/ADT/TypeSwitch.h"

#define GET_OP_CLASSES
#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPack.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackTypes.cpp.inc"

namespace libtriton::dlpack {

void DLPackDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPack.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackTypes.cpp.inc"
      >();
}

} // namespace libtriton::dlpack
