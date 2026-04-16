add_library(LibTritonCoreRuntimeCPU SHARED
  CPUMemRefRuntime.c
  ExecutionEngineRuntimeExports.cc
)

target_link_libraries(LibTritonCoreRuntimeCPU PUBLIC
  tvm_ffi::header
  LLVMSupport
)

find_package(CUDAToolkit REQUIRED)

add_library(LibTritonCoreRuntimeCUDA SHARED
  CUDAMemRefRuntime.c
  ExecutionEngineRuntimeExports.cc
)

target_link_libraries(LibTritonCoreRuntimeCUDA PUBLIC
  tvm_ffi::header
  CUDA::cudart
  CUDA::cuda_driver
  LLVMSupport
)

install(TARGETS
  LibTritonCoreRuntimeCPU
  LibTritonCoreRuntimeCUDA
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
