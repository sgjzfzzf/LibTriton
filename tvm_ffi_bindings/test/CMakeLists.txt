execute_process(
  COMMAND which lit
  OUTPUT_VARIABLE LLVM_EXTERNAL_LIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(LLVM_EXTERNAL_LIT "${LLVM_EXTERNAL_LIT}" CACHE STRING "Command used to spawn lit" FORCE)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TVM_FFI_BINDINGS_TEST_DEPENDS
  FileCheck
  tvm-ffi-opt
)

add_lit_testsuite(check-tvm-ffi-bindings "Running the TVM FFI bindings regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TVM_FFI_BINDINGS_TEST_DEPENDS}
)

set_target_properties(check-tvm-ffi-bindings PROPERTIES FOLDER "Tests")

add_lit_testsuites(TVM_FFI_BINDINGS ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${TVM_FFI_BINDINGS_TEST_DEPENDS}
)
