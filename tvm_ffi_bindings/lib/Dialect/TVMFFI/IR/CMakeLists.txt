add_mlir_dialect_library(TVMFFIDialect
  TVMFFIDialect.cc
  DEPENDS
  MLIRTVMFFIIncGen
)
