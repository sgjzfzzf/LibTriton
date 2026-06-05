include "mlir/Pass/PassBase.td"

def ConvertDLPackToLLVM : Pass<"convert-dlpack-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower DLPack dialect operations to LLVM dialect";
  let description = [{Lower DLPack dialect operations to LLVM dialect.}];
}

def ConvertTVMFFIToLLVM : Pass<"convert-tvm-ffi-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower TVMFFI dialect operations to LLVM dialect";
  let description = [{Lower TVMFFI dialect operations to LLVM dialect.}];
}

def ConvertTorchToCf : Pass<"convert-torch-to-cf", "mlir::ModuleOp"> {
  let summary = "Lower Torch dialect runtime ops to ControlFlow dialect";
  let description = [{Lower Torch dialect runtime ops to ControlFlow dialect.}];
  let dependentDialects = ["mlir::cf::ControlFlowDialect"];
}

def ConvertTorchToLLVM : Pass<"convert-torch-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower Torch dialect operations to LLVM dialect";
  let description = [{Lower Torch dialect operations to LLVM dialect.}];
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

def ConvertTorchExtToLLVM : Pass<"convert-torchext-to-llvm", "mlir::ModuleOp"> {
  let summary =
      "Lower TorchExt launch and runtime ops to GPU/LLVM-compatible form";
  // clang-format off
  let description = [{
    Rewrites torch_ext.triton_kernel_launch into gpu.launch_func while keeping
      async launch semantics, and lowers torch_ext.get_current_device and
      torch_ext.get_current_stream into LibTriton runtime C API calls.
  }];
  // clang-format on
  let dependentDialects = [
    "libtriton::torch_ext::TorchExtDialect", "mlir::func::FuncDialect",
    "mlir::gpu::GPUDialect", "mlir::LLVM::LLVMDialect"
  ];
}
