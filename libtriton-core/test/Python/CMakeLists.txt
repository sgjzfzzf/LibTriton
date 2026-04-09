find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(_libtriton_core_python_package_dir
  ${CMAKE_BINARY_DIR}/python_packages/libtriton-core
)

get_filename_component(_mlir_install_root "${MLIR_DIR}/../../.." ABSOLUTE)
set(_mlir_python_package_dir
  ${_mlir_install_root}/python_packages/mlir_core
)

add_test(
  NAME libtriton-core-python-smoke
  COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH=${_libtriton_core_python_package_dir}:${_mlir_python_package_dir}
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/smoke.py
)

set_tests_properties(libtriton-core-python-smoke PROPERTIES
  ENVIRONMENT "PYTHONPATH=${_libtriton_core_python_package_dir}:${_mlir_python_package_dir}"
)
