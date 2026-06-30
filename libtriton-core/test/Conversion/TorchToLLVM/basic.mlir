// RUN: libtriton-core-opt %s --convert-torch-to-llvm | FileCheck %s
//
// Tests that the standalone ConvertTorchToLLVM pass lowers
// torch.constant.bool/int/float/none to TVMFFIAny structs.
// Function signatures remain unchanged (type conversion is handled by the
// separate func-backend-type-conversion pass).

// CHECK-LABEL:   func.func @torch.constant.bool() -> !torch.bool {
// CHECK:           %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[IDX:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[WITH_IDX:.*]] = llvm.insertvalue %[[IDX]], %[[UNDEF]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[PAYLOAD:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[WITH_PLD:.*]] = llvm.insertvalue %[[PAYLOAD]], %[[WITH_IDX]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[C:.*]] = builtin.unrealized_conversion_cast %[[WITH_PLD]] : !llvm.struct<(i32, i32, i64)> to !torch.bool
// CHECK-NEXT:      return %[[C]] : !torch.bool
func.func @torch.constant.bool() -> !torch.bool {
  %true = torch.constant.bool true
  return %true : !torch.bool
}

// CHECK-LABEL:   func.func @torch.constant.int() -> !torch.int {
// CHECK:           %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[IDX:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[WITH_IDX:.*]] = llvm.insertvalue %[[IDX]], %[[UNDEF]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[PAYLOAD:.*]] = llvm.mlir.constant(42 : i64) : i64
// CHECK:           %[[WITH_PLD:.*]] = llvm.insertvalue %[[PAYLOAD]], %[[WITH_IDX]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[C:.*]] = builtin.unrealized_conversion_cast %[[WITH_PLD]] : !llvm.struct<(i32, i32, i64)> to !torch.int
// CHECK-NEXT:      return %[[C]] : !torch.int
func.func @torch.constant.int() -> !torch.int {
  %int = torch.constant.int 42
  return %int : !torch.int
}

// CHECK-LABEL:   func.func @torch.constant.float() -> !torch.float {
// CHECK:           %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[IDX:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:           %[[WITH_IDX:.*]] = llvm.insertvalue %[[IDX]], %[[UNDEF]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[FVAL:.*]] = llvm.mlir.constant(3.140000e+00 : f64) : f64
// CHECK:           %[[BC:.*]] = llvm.bitcast %[[FVAL]] : f64 to i64
// CHECK:           %[[WITH_PLD:.*]] = llvm.insertvalue %[[BC]], %[[WITH_IDX]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[C:.*]] = builtin.unrealized_conversion_cast %[[WITH_PLD]] : !llvm.struct<(i32, i32, i64)> to !torch.float
// CHECK-NEXT:      return %[[C]] : !torch.float
func.func @torch.constant.float() -> !torch.float {
  %float = torch.constant.float 3.14
  return %float : !torch.float
}

// CHECK-LABEL:   func.func @torch.constant.none() -> !torch.none {
// CHECK:           %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[IDX:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[WITH_IDX:.*]] = llvm.insertvalue %[[IDX]], %[[UNDEF]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[PAYLOAD:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[WITH_PLD:.*]] = llvm.insertvalue %[[PAYLOAD]], %[[WITH_IDX]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[C:.*]] = builtin.unrealized_conversion_cast %[[WITH_PLD]] : !llvm.struct<(i32, i32, i64)> to !torch.none
// CHECK-NEXT:      return %[[C]] : !torch.none
func.func @torch.constant.none() -> !torch.none {
  %none = torch.constant.none
  return %none : !torch.none
}
