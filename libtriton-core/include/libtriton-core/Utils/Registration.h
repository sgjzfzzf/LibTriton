#ifndef LIBTRITON_CORE_UTILS_REGISTRATION_H_
#define LIBTRITON_CORE_UTILS_REGISTRATION_H_

#include "mlir/IR/DialectRegistry.h"

namespace libtriton::conversion {

void registerAllPasses();
void registerAllDialects(mlir::DialectRegistry &registry);

} // namespace libtriton::conversion

#endif // LIBTRITON_CORE_UTILS_REGISTRATION_H_