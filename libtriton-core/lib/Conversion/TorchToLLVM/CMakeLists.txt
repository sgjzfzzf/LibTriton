add_mlir_conversion_library(
  TorchToLLVM
  TorchToLLVM.cc
  DEPENDS
  MLIRConversionPassIncGen
  LINK_LIBS
  PUBLIC
  MLIRPass)
