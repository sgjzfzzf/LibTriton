#ifndef LIBTRITON_CORE_DIALECT_AOTINDUCTOR_TRANSFORMS_REWRITORCHASAOTI_H_
#define LIBTRITON_CORE_DIALECT_AOTINDUCTOR_TRANSFORMS_REWRITORCHASAOTI_H_

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace libtriton::aoti {

#define GEN_PASS_DECL_REWRITETORCHASAOTI
#include "libtriton-core/Dialect/AOTInductor/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION_REWRITETORCHASAOTI
#include "libtriton-core/Dialect/AOTInductor/Transforms/Passes.h.inc"

void populateRewriteTorchAsAOTIPatterns(mlir::RewritePatternSet &patterns);

void registerRewriteTorchAsAOTIPass();

} // namespace libtriton::aoti

#endif // LIBTRITON_CORE_DIALECT_AOTINDUCTOR_TRANSFORMS_REWRITORCHASAOTI_H_
