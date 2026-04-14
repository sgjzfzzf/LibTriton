add_library(LibTritonCoreRuntime SHARED
  DLPackRuntime.c
)

target_link_libraries(LibTritonCoreRuntime PUBLIC
  tvm_ffi::header
)

install(TARGETS LibTritonCoreRuntime
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
