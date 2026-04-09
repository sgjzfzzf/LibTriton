set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

set(LIBTRITON_CORE_PYTHON_PACKAGE_NAME "libtriton-core")
set(LIBTRITON_CORE_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtriton_core")
set(LIBTRITON_CORE_PYTHON_PACKAGES_DIR "${CMAKE_BINARY_DIR}/python_packages")

declare_mlir_python_sources(LibTritonCorePythonSources)

declare_mlir_python_sources(LibTritonCorePythonSources.Core
  ROOT_DIR "${LIBTRITON_CORE_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT LibTritonCorePythonSources
  SOURCES
    __init__.py
  SOURCES_GLOB
    dialects/*.py
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT LibTritonCorePythonSources.Core
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/DLPackOpsPyBind.td
  DIALECT_NAME dlpack
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT LibTritonCorePythonSources.Core
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/TVMFFIOpsPyBind.td
  DIALECT_NAME tvm_ffi
)

add_mlir_python_sources_target(LibTritonCorePythonSourcesTarget
  INSTALL_COMPONENT LibTritonCorePythonModules
  INSTALL_DIR python_packages/${LIBTRITON_CORE_PYTHON_PACKAGE_NAME}/libtriton_core
  OUTPUT_DIRECTORY ${LIBTRITON_CORE_PYTHON_PACKAGES_DIR}/${LIBTRITON_CORE_PYTHON_PACKAGE_NAME}/libtriton_core
  SOURCES_TARGETS
    LibTritonCorePythonSources.Core
    LibTritonCorePythonSources.Core.dlpack.ops_gen
    LibTritonCorePythonSources.Core.tvm_ffi.ops_gen
)

build_nanobind_lib(
  INSTALL_COMPONENT LibTritonCorePythonModules
  INSTALL_DESTINATION python_packages/${LIBTRITON_CORE_PYTHON_PACKAGE_NAME}/libtriton_core
  OUTPUT_DIRECTORY ${LIBTRITON_CORE_PYTHON_PACKAGES_DIR}/${LIBTRITON_CORE_PYTHON_PACKAGE_NAME}/libtriton_core
  MLIR_BINDINGS_PYTHON_NB_DOMAIN libtriton_core
)

add_mlir_python_extension(LibTritonCorePythonExtension
  _libtritonCore
  ${NB_LIBRARY_TARGET_NAME}
  INSTALL_COMPONENT LibTritonCorePythonModules
  INSTALL_DIR python_packages/${LIBTRITON_CORE_PYTHON_PACKAGE_NAME}/libtriton_core
  OUTPUT_DIRECTORY ${LIBTRITON_CORE_PYTHON_PACKAGES_DIR}/${LIBTRITON_CORE_PYTHON_PACKAGE_NAME}/libtriton_core
  MLIR_BINDINGS_PYTHON_NB_DOMAIN libtriton_core
  SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/LibTritonCoreModule.cpp
  LINK_LIBS PRIVATE
    LibTritonCoreCAPI
    MLIRCAPIIR
    LLVMSupport
)
mlir_python_setup_extension_rpath(LibTritonCorePythonExtension)

add_custom_target(LibTritonCorePythonModules ALL
  DEPENDS
    LibTritonCorePythonSourcesTarget
    LibTritonCorePythonExtension
)

if(NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(install-LibTritonCorePythonModules
    DEPENDS LibTritonCorePythonModules
    COMPONENT LibTritonCorePythonModules
  )
endif()
