# SchemaLookup library: encapsulates schema lookup logic that calls into
# c10::Dispatcher, which requires C++ exceptions and RTTI.
add_library(SchemaLookup STATIC SchemaLookup.cc)
target_link_libraries(
  SchemaLookup
  PUBLIC MLIRLLVMDialect
  PRIVATE ${TORCH_LIBRARIES})

add_mlir_conversion_library(
  TorchExtToLLVM
  TorchExtToLLVM.cc
  DEPENDS
  MLIRConversionPassIncGen
  LINK_LIBS
  PUBLIC
  TorchExtDialect
  BackendTypeConversion
  ConversionUtils
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRPass
  PRIVATE
  SchemaLookup)
