//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIATTRIBUTES_H_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIATTRIBUTES_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"

#include "trident-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"

#define GET_ATTRDEF_CLASSES
#include "trident-core/Dialect/TVMFFI/IR/TVMFFIAttributes.h.inc"

#endif // TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIATTRIBUTES_H_
