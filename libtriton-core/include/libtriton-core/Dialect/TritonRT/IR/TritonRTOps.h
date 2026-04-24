#ifndef LIBTRITON_CORE_DIALECT_TRITONRT_IR_TRITONRTOPS_H_
#define LIBTRITON_CORE_DIALECT_TRITONRT_IR_TRITONRTOPS_H_

#include "libtriton-core/Dialect/TritonRT/IR/TritonRTTypes.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

#define GET_OP_CLASSES
#include "libtriton-core/Dialect/TritonRT/IR/TritonRT.h.inc"

#endif // LIBTRITON_CORE_DIALECT_TRITONRT_IR_TRITONRTOPS_H_
