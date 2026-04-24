#ifndef LIBTRITON_CORE_DIALECT_TRITONRT_IR_TRITONRTBASE_TD_
#define LIBTRITON_CORE_DIALECT_TRITONRT_IR_TRITONRTBASE_TD_

include "mlir/IR/DialectBase.td"

def TritonRT_Dialect : Dialect {
  let name = "triton_rt";
  let cppNamespace = "::libtriton::triton_rt";
  let useDefaultTypePrinterParser = 0;
  let summary = "Triton runtime dialect";
  let description = [{The TritonRT dialect models runtime -
                      facing Triton kernel launch semantics and related ABI -
                      level constructs.}];
}

#endif // LIBTRITON_CORE_DIALECT_TRITONRT_IR_TRITONRTBASE_TD_
