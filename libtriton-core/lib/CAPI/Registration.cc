#include "libtriton-core-c/Registration.h"

#include "libtriton-core/Utils/Registration.h"
#include "mlir/CAPI/IR.h"
#include "torch-mlir-c/Registration.h"

void libtritonCoreRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  libtriton::conversion::registerAllDialects(registry);
  unwrap(context)->appendDialectRegistry(registry);
  torchMlirRegisterAllDialects(context);
  unwrap(context)->loadAllAvailableDialects();
}

void libtritonCoreRegisterAllPasses(void) {
  libtriton::conversion::registerAllPasses();
}
