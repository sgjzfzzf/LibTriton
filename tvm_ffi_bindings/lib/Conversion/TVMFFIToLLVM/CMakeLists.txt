add_mlir_conversion_library(TVMFFIToLLVM
	TVMFFIToLLVM.cpp

	LINK_LIBS PUBLIC
	MLIRPass
	TVMFFIDialect
)
