#ifndef LIBTRITON_CORE_DIALECT_TORCHEXT_TRANSFORMS_PASSES_TD_
#define LIBTRITON_CORE_DIALECT_TORCHEXT_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def RAAI : Pass<"raai", "mlir::func::FuncOp"> {
  let summary = "Release allocated intermediate variables in single-region ops";
  // clang-format off
  let description = [{
    Tracks allocations (torch.prim.ListConstruct and torch.aten.* ops)
    inside func::FuncOp regions and inserts torchext.ObjectDecRef for all
    tracked values at every return point. Values that are returned (yielded)
    also receive a torchext.ObjectIncRef to transfer ownership to the caller.
  }];
  // clang-format on
  let dependentDialects =
      ["mlir::func::FuncDialect", "libtriton::torchext::TorchExtDialect"];
}

#endif // LIBTRITON_CORE_DIALECT_TORCHEXT_TRANSFORMS_PASSES_TD_
