#ifndef TVM_FFI_BINDINGS_DIALECT_TVMFFI_IR_TVMFFI_TD_
#define TVM_FFI_BINDINGS_DIALECT_TVMFFI_IR_TVMFFI_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def TVMFFI_Dialect : Dialect {
  let name = "tvm_ffi";
  let cppNamespace = "::libtriton::tvm_ffi";
  let useDefaultTypePrinterParser = 1;
  let summary = "TVM runtime FFI modeling dialect";
  let description =
      [{The TVMFFI dialect models value conversions between MLIR -
        level values and runtime FFI -
        compatible handles used by the TVM runtime boundary
            .

        At this stage the dialect primarily provides conversion scaffold
            operations and custom types used by lowering passes.}];
}

include "tvm_ffi_bindings/Dialect/TVMFFI/IR/TVMFFITypes.td"
include "tvm_ffi_bindings/Dialect/TVMFFI/IR/TVMFFIOps.td"

#endif // TVM_FFI_BINDINGS_DIALECT_TVMFFI_IR_TVMFFI_TD_
