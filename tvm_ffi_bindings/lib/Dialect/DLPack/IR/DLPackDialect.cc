#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackDialect.cpp.inc"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackTypes.cpp.inc"

namespace libtriton::dlpack {

void DLPackDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackTypes.cpp.inc"
      >();
}

} // namespace libtriton::dlpack
