#ifndef LIBTRITON_CORE_DIALECT_AOTINDUCTOR_IR_AOTINDUCTOROPS_H_
#define LIBTRITON_CORE_DIALECT_AOTINDUCTOR_IR_AOTINDUCTOROPS_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

#define GET_OP_CLASSES
#include "libtriton-core/Dialect/AOTInductor/IR/AOTInductor.h.inc"

#endif // LIBTRITON_CORE_DIALECT_AOTINDUCTOR_IR_AOTINDUCTOROPS_H_
