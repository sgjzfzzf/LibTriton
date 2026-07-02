#ifndef TRIDENT_CORE_UTILS_REGISTRATION_H_
#define TRIDENT_CORE_UTILS_REGISTRATION_H_

#include "mlir/IR/DialectRegistry.h"

namespace trident::conversion {

void registerAllPasses();
void registerAllDialects(mlir::DialectRegistry &registry);

} // namespace trident::conversion

#endif // TRIDENT_CORE_UTILS_REGISTRATION_H_