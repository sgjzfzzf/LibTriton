add_mlir_library(
  BackendTypeConversion
  BackendTypeConversion.cc
  LINK_LIBS
  PUBLIC
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRPass
  TorchMLIRTorchDialect)

add_mlir_library(
  TorchExtBufferize
  BufferizableOpInterfaceImpl.cc
  LINK_LIBS
  PUBLIC
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  TorchExtDialect)
