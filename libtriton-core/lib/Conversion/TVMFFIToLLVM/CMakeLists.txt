add_mlir_conversion_library(TVMFFIToLLVM
	TVMFFIToLLVM.cc

	DEPENDS
	MLIRTVMFFIToLLVMPassIncGen

	LINK_LIBS PUBLIC
	MemRefOriginAnalysis
	MLIRPass
	TVMFFIDialect
	tvm_ffi::header
)
