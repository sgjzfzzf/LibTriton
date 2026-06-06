// RUN: libtriton-core-opt %s --torch-to-llvm-pipeline | FileCheck %s
//
// This test verifies that the TorchToLLVM pipeline correctly lowers
// torch.aten.empty_like (with torch.constant.bool) all the way to LLVM
// dialect.  torch.constant.* ops are lowered by ConvertTorchToLLVM
// (instead of the old TorchToArith pass).

// CHECK:       llvm.mlir.global internal constant @__aoti_op_name_overload_
// CHECK:       llvm.mlir.global internal constant @"__aoti_op_name_op_aten::empty_like"
// CHECK:       llvm.func @aoti_torch_call_dispatcher
// CHECK-SAME:    (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

// After ConvertTorchToLLVM + ConvertToLLVM the function becomes llvm.func.
// CHECK-LABEL: llvm.func @torch.aten.empty_like(
// CHECK-SAME:    %[[ARG0:.*]]: !llvm.ptr) -> !llvm.ptr {
// No arith.constant or torch_c.from_i1 should remain.
// CHECK-NOT:     arith.constant
// CHECK-NOT:     torch_c.from_i1
// The pipeline must produce LLVM constants directly.
// CHECK:         %[[OVERLOAD:.*]] = llvm.mlir.addressof @__aoti_op_name_overload_
// CHECK-SAME:      : !llvm.ptr
// CHECK:         %[[OPNAME:.*]] = llvm.mlir.addressof @"__aoti_op_name_op_aten::empty_like"
// CHECK-SAME:      : !llvm.ptr
// The bool constant false was lowered to i1, zext‑folded to i64 0, then
// CSE'd with the other zero constants.
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[C6:.*]] = llvm.mlir.constant(6 : i64) : i64
// CHECK-NEXT:    %[[SLOTS:.*]] = llvm.alloca %[[C6]] x i64 : (i64) -> !llvm.ptr
// Store arg0 ptr, then 5 zero slots for 5 none/bool args.
// CHECK:         %{{.*}} = llvm.ptrtoint %[[ARG0]] : !llvm.ptr to i64
// CHECK:         llvm.store
// CHECK:         llvm.store
// CHECK:         llvm.store
// CHECK:         llvm.store
// CHECK:         llvm.store
// CHECK:         llvm.store
// CHECK:         %[[RET:.*]] = llvm.call @aoti_torch_call_dispatcher(%[[OPNAME]], %[[OVERLOAD]], %[[SLOTS]])
// CHECK-SAME:      : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         %[[RES_VAL:.*]] = llvm.load
// CHECK:         %[[RES_PTR:.*]] = llvm.inttoptr
// CHECK-NEXT:    llvm.return %[[RES_PTR]] : !llvm.ptr
func.func @torch.aten.empty_like(%arg0: !torch.vtensor<[200,200,26],f64>) -> !torch.vtensor<[200,200,26],f64> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten.empty_like %arg0, %none, %none, %none, %false, %none : !torch.vtensor<[200,200,26],f64>, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[200,200,26],f64>
  return %0 : !torch.vtensor<[200,200,26],f64>
}
