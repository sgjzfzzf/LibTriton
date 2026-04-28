find_package(CUDAToolkit REQUIRED)

add_library(LibTritonCoreRuntime SHARED Runtime.cc)

target_link_libraries(LibTritonCoreRuntime PUBLIC tvm_ffi::header CUDA::cudart
                                                  CUDA::cuda_driver LLVMSupport)

install(
  TARGETS LibTritonCoreRuntime
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
