//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBTRITON_CORE_CONVERSION_TORCHTOLLVM_BACKENDTYPECONVERSION_H_
#define LIBTRITON_CORE_CONVERSION_TORCHTOLLVM_BACKENDTYPECONVERSION_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class LLVMTypeConverter;
} // namespace mlir

namespace libtriton::torch {

/// Set up the provided ConversionTarget and LLVMTypeConverter for converting
/// from Torch dialect types to LLVM types along the backend boundary.
/// Currently handles:
///   - Torch tensor types (ValueTensorType, NonValueTensorType) -> llvm.ptr
///   - Torch BoolType -> i1
///   - Torch IntType -> i64
///   - Torch FloatType -> f64
/// Generator conversion is not implemented yet.
void setupBackendTypeConversion(mlir::ConversionTarget &target,
                                mlir::LLVMTypeConverter &typeConverter);

} // namespace libtriton::torch

#endif // LIBTRITON_CORE_CONVERSION_TORCHTOLLVM_BACKENDTYPECONVERSION_H_
