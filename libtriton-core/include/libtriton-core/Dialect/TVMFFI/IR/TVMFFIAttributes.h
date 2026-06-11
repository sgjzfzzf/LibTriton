//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIATTRIBUTES_H_
#define LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIATTRIBUTES_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"

#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"

#define GET_ATTRDEF_CLASSES
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIAttributes.h.inc"

#endif // LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIATTRIBUTES_H_
