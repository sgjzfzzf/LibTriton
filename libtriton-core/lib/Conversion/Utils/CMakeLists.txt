add_mlir_conversion_library(
  ConversionUtils
  GlobalString.cc
  Type.cc
  LINK_LIBS
  PUBLIC
  MLIRLLVMDialect
  MLIRTransforms)
