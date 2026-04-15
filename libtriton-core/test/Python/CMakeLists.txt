find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(_libtriton_core_python_package_dir
  ${CMAKE_BINARY_DIR}/python_packages/libtriton-core
)

set(_libtriton_python_source_dir
  ${CMAKE_SOURCE_DIR}/../python
)

add_test(
  NAME libtriton-core-python-unittest
  COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH=${_libtriton_python_source_dir}:${_libtriton_core_python_package_dir}
    ${Python3_EXECUTABLE}
    ${_libtriton_python_source_dir}/libtriton/tests/run_tests.py
)

set_tests_properties(libtriton-core-python-unittest PROPERTIES
  ENVIRONMENT "PYTHONPATH=${_libtriton_python_source_dir}:${_libtriton_core_python_package_dir}"
)
