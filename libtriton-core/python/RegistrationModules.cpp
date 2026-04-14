#include <cstdint>

#include "libtriton-core-c/Dialects.h"
#include "libtriton-core-c/Registration.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "torch-mlir-c/Dialects.h"
#include "torch-mlir-c/Registration.h"

namespace nb = nanobind;

namespace {

void registerDialect(MlirDialectHandle handle, MlirContext context, bool load) {
  mlirDialectHandleRegisterDialect(handle, context);
  if (load) {
    mlirDialectHandleLoadDialect(handle, context);
  }
}

void registerLibTritonDialects(MlirContext context, bool load) {
  registerDialect(mlirGetDialectHandle__dlpack__(), context, load);
  registerDialect(mlirGetDialectHandle__tvm_ffi__(), context, load);
  registerDialect(mlirGetDialectHandle__torch__(), context, load);
}

void registerAllDialects(MlirContext context, bool load) {
  if (load) {
    libtritonCoreRegisterAllDialects(context);
    return;
  }
  registerLibTritonDialects(context, false);
}

void registerAllPasses() { libtritonCoreRegisterAllPasses(); }

} // namespace

NB_MODULE(_libtritonCore, m) {
  m.doc() = "libtriton-core python extension";

  m.def("register_all_dialects", &registerAllDialects, nb::arg("context"),
        nb::arg("load") = true);
  m.def("register_all_passes", &registerAllPasses);
}

NB_MODULE(_torchMlirCore, m) {
  torchMlirRegisterAllPasses();

  m.doc() = "libtriton-core torch registration extension";

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        registerDialect(mlirGetDialectHandle__torch__(), context, load);
      },
      nb::arg("context"), nb::arg("load") = true);
  m.def("get_int64_max", []() { return INT64_MAX; });
  m.def("get_int64_min", []() { return INT64_MIN; });
}
