include "mlir/Pass/PassBase.td"

def NormalizeTritonRTOperands
    : Pass<"tritonrt-normalize-operands", "mlir::func::FuncOp"> {
  let summary =
      "Canonicalize TritonRT launch operands to builtin tensor values";
  let description =
      [{Rewrites triton_rt.triton_kernel_launch operands produced by torch_c
            .to_builtin_tensor to use the underlying builtin tensor values.}];
  let dependentDialects = [
    "libtriton::triton_rt::TritonRTDialect", "mlir::func::FuncDialect",
    "mlir::torch::TorchConversion::TorchConversionDialect"
  ];
}
