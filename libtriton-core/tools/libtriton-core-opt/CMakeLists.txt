add_mlir_tool(libtriton-core-opt libtriton-core-opt.cc)

set_target_properties(libtriton-core-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                    "${PROJECT_BINARY_DIR}/bin")

target_link_libraries(
  libtriton-core-opt
  PRIVATE TorchExtDialect
          LibTritonUtilsRegistration
          MLIRMlirOptMain
          TVMFFIDialect
          "${Python3_SITELIB}/nvidia/cuda_runtime/lib/libcudart.so.12"
          "${Python3_SITELIB}/nvidia/cuda_cupti/lib/libcupti.so.12")

mlir_check_all_link_libraries(libtriton-core-opt)
