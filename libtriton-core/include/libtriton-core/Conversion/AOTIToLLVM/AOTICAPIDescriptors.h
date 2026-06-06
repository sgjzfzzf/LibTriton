#ifndef LIBTRITON_CORE_CONVERSION_AOTITOLLVM_AOTICAPIDESCRIPTORS_H_
#define LIBTRITON_CORE_CONVERSION_AOTITOLLVM_AOTICAPIDESCRIPTORS_H_

#include "libtriton-core/Conversion/Utils/CFunctionDeclUtils.h"
#include "torch/csrc/inductor/aoti_torch/c/shim.h"

namespace libtriton::aoti::capi {

LIBTRITON_DECLARE_CAPI_GET_OR_CREATE(aoti_torch_call_dispatcher)

} // namespace libtriton::aoti::capi

#endif // LIBTRITON_CORE_CONVERSION_AOTITOLLVM_AOTICAPIDESCRIPTORS_H_
