// tvm-ffi-opt.cpp - Standalone optimizer driver for the TVMFFI dialect.
//
// Registers TVMFFI conversion passes and inserts the TVMFFIDialect,
// func::FuncDialect, and LLVM::LLVMDialect into the registry so that
// tvm-ffi-opt can parse and transform `.mlir` files exercising the dialect.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "tvm_ffi_bindings/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "tvm_ffi_bindings/Dialect/TVMFFI/IR/TVMFFIDialect.h"

int main(int argc, char **argv) {
  libtriton::tvm_ffi::registerTVMFFIToLLVMPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                  libtriton::tvm_ffi::TVMFFIDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "TVM FFI modular optimizer driver\n", registry));
}
