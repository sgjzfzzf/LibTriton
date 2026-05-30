include "mlir/Pass/PassBase.td"

def NormalizeTorchExtOperands
    : Pass<"torchext-normalize-operands", "mlir::func::FuncOp"> {
  let summary =
      "Canonicalize TorchExt launch operands to builtin tensor values";
  let description =
      [{Rewrites torch_ext.triton_kernel_launch operands produced by torch_c
            .to_builtin_tensor to use the underlying builtin tensor values.}];
  let dependentDialects = [
    "libtriton::torch_ext::TorchExtDialect", "mlir::func::FuncDialect",
    "mlir::torch::TorchConversion::TorchConversionDialect"
  ];
}

def AsyncKernelLaunch : Pass<"torchext-async-kernel-launch", "mlir::ModuleOp"> {
  let summary = "Materialize current-stream async tokens for TorchExt";
  // clang-format off
  let description = [{
    Rewrites gpu.launch_func and torch_ext.triton_kernel_launch in a module.
    If launch async information is not provided, this pass inserts
    torch_ext.get_current_stream and wires the produced !gpu.async.token as an
    async dependency.
  }];
  // clang-format on
  let dependentDialects =
      ["libtriton::torch_ext::TorchExtDialect", "mlir::gpu::GPUDialect"];
}
