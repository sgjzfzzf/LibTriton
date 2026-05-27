#ifndef LIBTRITON_CORE_CONVERSION_BUFFERIZATION_BUFFERIZATION_H_
#define LIBTRITON_CORE_CONVERSION_BUFFERIZATION_BUFFERIZATION_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace libtriton::conversion {

#define GEN_PASS_DECL_LIBTRITONONESHOTBUFFERIZE
#include "libtriton-core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_LIBTRITONONESHOTBUFFERIZE
#include "libtriton-core/Conversion/Passes.h.inc"

void registerLibTritonOneShotBufferizePass();

} // namespace libtriton::conversion

#endif // LIBTRITON_CORE_CONVERSION_BUFFERIZATION_BUFFERIZATION_H_
