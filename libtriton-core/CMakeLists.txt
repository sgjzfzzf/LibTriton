cmake_minimum_required(VERSION 3.29)

project(LibTritonCore)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(MLIR REQUIRED CONFIG)
find_package(tvm_ffi CONFIG REQUIRED
  PATHS
  ${Python3_SITELIB}/tvm_ffi/share/cmake/tvm_ffi
  ${Python3_SITEARCH}/tvm_ffi/share/cmake/tvm_ffi
)

set(LIBTRITON_CORE_ENABLE_PYTHON_BINDINGS ON CACHE BOOL "Enable libtriton-core Python bindings")

include(${LLVM_CMAKE_DIR}/AddLLVM.cmake)
include(${MLIR_CMAKE_DIR}/AddMLIR.cmake)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

if(LIBTRITON_CORE_ENABLE_PYTHON_BINDINGS)
  find_package(Python REQUIRED COMPONENTS Interpreter Development)
  find_package(nanobind CONFIG REQUIRED
    PATHS
    ${Python_SITELIB}/nanobind/cmake
    ${Python_SITEARCH}/nanobind/cmake
  )
  include(${MLIR_CMAKE_DIR}/AddMLIRPython.cmake)
  add_subdirectory(python)
endif()

add_subdirectory(test)
