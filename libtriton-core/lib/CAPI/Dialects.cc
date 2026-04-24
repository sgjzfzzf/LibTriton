#include "libtriton-core-c/Dialects.h"

#include "libtriton-core/Dialect/DLPack/IR/DLPackDialect.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "libtriton-core/Dialect/TritonRT/IR/TritonRTDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(DLPack, dlpack,
                                      libtriton::dlpack::DLPackDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TritonRT, triton_rt,
                                      libtriton::triton_rt::TritonRTDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TVMFFI, tvm_ffi,
                                      libtriton::tvm_ffi::TVMFFIDialect)
