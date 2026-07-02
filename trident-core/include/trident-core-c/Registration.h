#ifndef TRIDENT_CORE_C_REGISTRATION_H
#define TRIDENT_CORE_C_REGISTRATION_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void tridentCoreRegisterAllDialects(MlirContext context);
MLIR_CAPI_EXPORTED void tridentCoreRegisterAllPasses(void);

#ifdef __cplusplus
}
#endif

#endif // TRIDENT_CORE_C_REGISTRATION_H
