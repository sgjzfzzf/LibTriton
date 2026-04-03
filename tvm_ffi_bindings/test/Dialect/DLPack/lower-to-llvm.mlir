// RUN: tvm-ffi-opt %s -convert-dlpack-to-llvm | FileCheck %s

// CHECK-LABEL: func.func @lower_from_memref
func.func @lower_from_memref(%arg0: memref<?xf32>) -> !dlpack.dltensor {
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : memref<?xf32> to !dlpack.dltensor
  %0 = dlpack.from_memref %arg0 : memref<?xf32> -> !dlpack.dltensor
  // CHECK: return %[[CAST]] : !dlpack.dltensor
  return %0 : !dlpack.dltensor
}

// -----

// CHECK-LABEL: func.func @lower_to_memref
func.func @lower_to_memref(%arg0: !dlpack.dltensor) -> memref<?xf32> {
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : !dlpack.dltensor to memref<?xf32>
  %0 = dlpack.to_memref %arg0 : !dlpack.dltensor -> memref<?xf32>
  // CHECK: return %[[CAST]] : memref<?xf32>
  return %0 : memref<?xf32>
}