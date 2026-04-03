// RUN: tvm-ffi-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @from_memref
func.func @from_memref(%arg0: memref<?xf32>) {
  // CHECK: %[[VALUE:.*]] = dlpack.from_memref %arg0 : memref<?xf32> -> !dlpack.dltensor
  %0 = dlpack.from_memref %arg0 : memref<?xf32> -> !dlpack.dltensor
  return
}

// -----

// CHECK-LABEL: func.func @to_memref
func.func @to_memref(%arg0: !dlpack.dltensor) {
  // CHECK: %[[VALUE:.*]] = dlpack.to_memref %arg0 : !dlpack.dltensor -> memref<?xf32>
  %0 = dlpack.to_memref %arg0 : !dlpack.dltensor -> memref<?xf32>
  return
}