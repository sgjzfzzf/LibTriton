add_mlir_library(TritonRTNormalizeOperands
  TritonRTNormalizeOperands.cc

  DEPENDS
  MLIRTritonRTPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRPass
  TorchMLIRTorchConversionDialect
  TorchMLIRTorchDialect
  TritonRTDialect
)
