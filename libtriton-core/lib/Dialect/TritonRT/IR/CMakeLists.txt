add_mlir_dialect_library(TritonRTDialect
  TritonRTDialect.cc
  DEPENDS
  MLIRTritonRTIncGen
  MLIRTorchTypesIncGen
  LINK_LIBS PUBLIC
  TorchMLIRTorchDialect
  TorchMLIRTorchUtils
)
