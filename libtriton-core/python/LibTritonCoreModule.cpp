#include "libtriton-core-c/DLPackTypes.h"
#include "libtriton-core-c/Dialects.h"
#include "libtriton-core-c/Registration.h"
#include "libtriton-core-c/TVMFFITypes.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

NB_MODULE(_libtritonCore, m) {
  m.doc() = "libtriton-core python extension";

  m.def(
      "register_all_dialects",
      [](MlirContext context, bool load) {
        MlirDialectHandle dlpackHandle = mlirGetDialectHandle__dlpack__();
        MlirDialectHandle tvmffiHandle = mlirGetDialectHandle__tvm_ffi__();

        mlirDialectHandleRegisterDialect(dlpackHandle, context);
        mlirDialectHandleRegisterDialect(tvmffiHandle, context);
        if (load) {
          mlirDialectHandleLoadDialect(dlpackHandle, context);
          mlirDialectHandleLoadDialect(tvmffiHandle, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);
  m.def("register_all_passes", &libtritonCoreRegisterAllPasses);

  m.def("dlpack_is_dl_context_type", &libtritonCoreTypeIsADLPackDLContextType,
        nb::arg("type"));
  m.def("dlpack_is_dl_data_type", &libtritonCoreTypeIsADLPackDLDataTypeType,
        nb::arg("type"));
  m.def("dlpack_is_dl_tensor_type", &libtritonCoreTypeIsADLPackDLTensorType,
        nb::arg("type"));
  m.def("dlpack_is_dl_managed_tensor_type",
        &libtritonCoreTypeIsADLPackDLManagedTensorType, nb::arg("type"));

  m.def("dlpack_get_dl_context_type", &libtritonCoreDLPackDLContextTypeGet,
        nb::arg("context"));
  m.def("dlpack_get_dl_data_type", &libtritonCoreDLPackDLDataTypeTypeGet,
        nb::arg("context"));
  m.def("dlpack_get_dl_tensor_type", &libtritonCoreDLPackDLTensorTypeGet,
        nb::arg("context"));
  m.def("dlpack_get_dl_managed_tensor_type",
        &libtritonCoreDLPackDLManagedTensorTypeGet, nb::arg("context"));

  m.def("tvm_ffi_is_any_type", &libtritonCoreTypeIsATVMFFIAnyType,
        nb::arg("type"));
  m.def("tvm_ffi_is_object_handle_type",
        &libtritonCoreTypeIsATVMFFIObjectHandleType, nb::arg("type"));

  m.def("tvm_ffi_get_any_type", &libtritonCoreTVMFFIAnyTypeGet,
        nb::arg("context"));
  m.def("tvm_ffi_get_object_handle_type",
        &libtritonCoreTVMFFIObjectHandleTypeGet, nb::arg("context"));
}
