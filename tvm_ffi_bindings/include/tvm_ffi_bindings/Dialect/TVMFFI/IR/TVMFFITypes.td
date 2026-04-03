#ifndef TVM_FFI_BINDINGS_DIALECT_TVMFFI_IR_TVMFFITYPES_TD_
#define TVM_FFI_BINDINGS_DIALECT_TVMFFI_IR_TVMFFITYPES_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "tvm_ffi_bindings/Dialect/TVMFFI/IR/TVMFFI.td"

class TVMFFI_Type<string name, string typeMnemonic>
    : TypeDef<TVMFFI_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def TVMFFI_AnyType : TVMFFI_Type<"any", "any"> {
  let summary = "Type-erased TVM FFI value";
  let description =
      [{A generic, type - erased container type used at the TVM FFI boundary
                              .Conversion ops box typed MLIR values
                                  into this type and unbox them later.}];
}

#endif // TVM_FFI_BINDINGS_DIALECT_TVMFFI_IR_TVMFFITYPES_TD_
