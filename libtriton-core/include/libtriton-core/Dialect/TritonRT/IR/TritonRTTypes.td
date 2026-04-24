#ifndef LIBTRITON_CORE_DIALECT_TRITONRT_IR_TRITONRTTYPES_TD_
#define LIBTRITON_CORE_DIALECT_TRITONRT_IR_TRITONRTTYPES_TD_

include "mlir/IR/AttrTypeBase.td"
include "libtriton-core/Dialect/TritonRT/IR/TritonRTBase.td"

class TritonRT_Type<string name, string typeMnemonic>
    : TypeDef<TritonRT_Dialect, name> {
  let mnemonic = typeMnemonic;
}

#endif // LIBTRITON_CORE_DIALECT_TRITONRT_IR_TRITONRTTYPES_TD_
