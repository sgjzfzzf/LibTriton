// libtriton-core-opt.cc - Standalone optimizer driver for TVMFFI and DLPack
// dialects.
//
// Registers core LLVM conversion pipeline passes and inserts dialects plus
// convert-to-LLVM interfaces needed by tests and development passes so
// libtriton-core-opt can parse and transform `.mlir` files exercising these
// dialects.

#include "libtriton-core/Utils/Registration.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  libtriton::conversion::registerAllPasses();

  mlir::DialectRegistry registry;
  libtriton::conversion::registerAllDialects(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "TVM FFI modular optimizer driver\n", registry));
}
