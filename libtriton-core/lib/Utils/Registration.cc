#include "libtriton-core/Utils/Registration.h"

#include "libtriton-core/Conversion/AOTIToLLVM/AOTIToLLVM.h"
#include "libtriton-core/Conversion/DLPackToLLVM/DLPackToLLVM.h"
#include "libtriton-core/Conversion/Pipeline/Pipeline.h"
#include "libtriton-core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "libtriton-core/Conversion/TorchConversionToLLVM/TorchConversionToLLVM.h"
#include "libtriton-core/Conversion/TorchExtToLLVM/TorchExtToLLVM.h"
#include "libtriton-core/Conversion/TorchToCf/TorchToCf.h"
#include "libtriton-core/Conversion/TorchToLLVM/FuncBackendTypeConversion.h"
#include "libtriton-core/Conversion/TorchToLLVM/TorchToLLVM.h"
#include "libtriton-core/Dialect/AOTInductor/IR/AOTInductorDialect.h"
#include "libtriton-core/Dialect/AOTInductor/Transforms/RewriteTorchAsAOTI.h"
#include "libtriton-core/Dialect/DLPack/IR/DLPackDialect.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "libtriton-core/Dialect/TVMFFI/Transforms/Passes.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "libtriton-core/Dialect/TorchExt/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir-c/Registration.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"

void libtriton::conversion::registerAllPasses() {
  mlir::registerAllPasses();
  mlir::registerConvertToLLVMPass();
  mlir::registerReconcileUnrealizedCastsPass();
  libtriton::aoti::registerConvertAOTIToLLVMPass();
  libtriton::aoti::registerRewriteTorchAsAOTIPass();
  libtriton::dlpack::registerConvertDLPackToLLVMPass();
  libtriton::torch_ext::registerConvertTorchExtToLLVMPass();
  libtriton::torch::registerConvertTorchToCfPass();
  libtriton::torch::registerFuncBackendTypeConversionPass();
  libtriton::torch::registerConvertTorchToLLVMPass();
  libtriton::torch::registerConvertTorchConversionToLLVMPass();
  libtriton::torch::registerTorchToLLVMPipelinePass();
  libtriton::tvm_ffi::registerConvertTVMFFIToLLVMPass();
  libtriton::tvm_ffi::registerFinalizeTVMFFICallPass();
}

void libtriton::conversion::registerAllDialects(
    mlir::DialectRegistry &registry) {
  mlir::registerAllDialects(registry);
  registry.insert<
      libtriton::aoti::AOTInductorDialect, libtriton::dlpack::DLPackDialect,
      libtriton::torch_ext::TorchExtDialect, libtriton::tvm_ffi::TVMFFIDialect,
      mlir::torch::Torch::TorchDialect,
      mlir::torch::TorchConversion::TorchConversionDialect>();
  mlir::registerAllExtensions(registry);
  libtriton::aoti::registerConvertAOTIToLLVMInterface(registry);
  libtriton::dlpack::registerConvertDLPackToLLVMInterface(registry);
  libtriton::torch::registerConvertTorchConversionToLLVMInterface(registry);
  libtriton::torch_ext::registerConvertTorchExtToLLVMInterface(registry);
  libtriton::tvm_ffi::registerConvertTVMFFIToLLVMInterface(registry);
  libtriton::torch_ext::registerBufferizableOpInterfaceExternalModels(registry);
}