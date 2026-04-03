#ifndef TVM_FFI_BINDINGS_DIALECT_TVMFFI_IR_TVMFFIDIALECT_H_
#define TVM_FFI_BINDINGS_DIALECT_TVMFFI_IR_TVMFFIDIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackDialect.h"
#include "tvm_ffi_bindings/Dialect/TVMFFI/IR/TVMFFIDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "tvm_ffi_bindings/Dialect/TVMFFI/IR/TVMFFITypes.h.inc"

#define GET_OP_CLASSES
#include "tvm_ffi_bindings/Dialect/TVMFFI/IR/TVMFFI.h.inc"

#endif // TVM_FFI_BINDINGS_DIALECT_TVMFFI_IR_TVMFFIDIALECT_H_
