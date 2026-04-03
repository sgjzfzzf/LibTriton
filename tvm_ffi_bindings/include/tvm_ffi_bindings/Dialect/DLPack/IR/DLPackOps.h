#ifndef TVM_FFI_BINDINGS_DIALECT_DLPACK_IR_DLPACKOPS_H_
#define TVM_FFI_BINDINGS_DIALECT_DLPACK_IR_DLPACKOPS_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackTypes.h"

#define GET_OP_CLASSES
#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPack.h.inc"

#endif // TVM_FFI_BINDINGS_DIALECT_DLPACK_IR_DLPACKOPS_H_