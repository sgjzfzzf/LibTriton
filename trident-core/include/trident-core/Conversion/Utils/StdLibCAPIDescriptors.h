//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_UTILS_STDLIBCAPIDESCRIPTORS_H_
#define TRIDENT_CORE_CONVERSION_UTILS_STDLIBCAPIDESCRIPTORS_H_

#include "trident-core/Conversion/Utils/CFunctionDeclUtils.h"
#include <cstdlib>

// Standard C library function descriptors.
namespace trident::conversion::utils {

TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(malloc, Malloc)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(free, Free)

} // namespace trident::conversion::utils

#endif // TRIDENT_CORE_CONVERSION_UTILS_STDLIBCAPIDESCRIPTORS_H_
