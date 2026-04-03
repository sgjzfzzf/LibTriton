// RUN: tvm-ffi-opt %s | FileCheck %s

// CHECK-LABEL: func.func @roundtrip_dltensor_type
// CHECK-SAME: (%[[ARG:.*]]: !dlpack.dltensor)
// CHECK: return %[[ARG]] : !dlpack.dltensor
func.func @roundtrip_dltensor_type(%arg0: !dlpack.dltensor) -> !dlpack.dltensor {
  return %arg0 : !dlpack.dltensor
}
