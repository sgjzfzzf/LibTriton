#ifndef LIBTRITON_CORE_DIALECT_AOTINDUCTOR_TRANSFORMS_PASSES_TD_
#define LIBTRITON_CORE_DIALECT_AOTINDUCTOR_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def RewriteTorchAsAOTI : Pass<"rewrite-torch-as-aoti", "mlir::ModuleOp"> {
  let summary =
      "Rewrite Torch dialect aten ops as AOTInductor dispatcher calls";
  let description = [{
    Rewrites Torch dialect ops whose names start with
        "torch.aten."(e.g.torch.aten.empty_like)
            into AOTInductor aoti.torch_call_dispatcher ops.The
        "aten." prefix becomes "aten::" as the dispatcher op_name;
    the overload_name is left empty.
  }];
  let dependentDialects = ["libtriton::aoti::AOTInductorDialect"];
}

#endif // LIBTRITON_CORE_DIALECT_AOTINDUCTOR_TRANSFORMS_PASSES_TD_
