#ifndef TVM_FFI_BINDINGS_DIALECT_DLPACK_IR_DLPACKOPS_TD_
#define TVM_FFI_BINDINGS_DIALECT_DLPACK_IR_DLPACKOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackBase.td"
include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackTypes.td"

class DLPack_Op<string mnemonic, list<Trait> traits = []>
    : Op<DLPack_Dialect, mnemonic, traits> {
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

def DLPack_DLTensorTypeConstraint
    : Type<CPred<"::llvm::isa<::libtriton::dlpack::DLTensorTypeType>($_self)">,
           "!dlpack.dltensor type">;

def DLPack_FromMemRefOp : DLPack_Op<"from_memref", [Pure]> {
  let summary = "Convert an MLIR memref value to a DLTensor handle";
  let description =
      [{Bridges a memref -
        typed MLIR value into a DLPack DLTensor handle value
            .This op is intended as an interoperability scaffold and can be
                lowered during conversion to LLVM dialect.}];
  let arguments = (ins AnyMemRef : $input);
  let results = (outs DLPack_DLTensorTypeConstraint : $output);
}

def DLPack_ToMemRefOp : DLPack_Op<"to_memref", [Pure]> {
  let summary = "Convert a DLTensor handle to an MLIR memref value";
  let description =
      [{Bridges a DLPack DLTensor handle value to an MLIR memref -
        typed value
            .This op is intended as an interoperability scaffold and can be
                lowered during conversion to LLVM dialect.}];
  let arguments = (ins DLPack_DLTensorTypeConstraint : $input);
  let results = (outs AnyMemRef : $output);
}

#endif // TVM_FFI_BINDINGS_DIALECT_DLPACK_IR_DLPACKOPS_TD_