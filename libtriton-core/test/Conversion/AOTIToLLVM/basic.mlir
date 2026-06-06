// RUN: libtriton-core-opt %s --rewrite-torch-as-aoti --convert-aoti-to-llvm --canonicalize | FileCheck %s

// CHECK-LABEL:   llvm.func @aoti_torch_call_dispatcher
// CHECK-SAME:      (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

// CHECK-LABEL:   func.func @torch.aten.empty_like(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr) -> !llvm.ptr {
// CHECK:           %[[OVERLOAD:.*]] = llvm.mlir.addressof @__aoti_op_name_overload_
// CHECK-SAME:        : !llvm.ptr
// CHECK:           %[[OPNAME:.*]] = llvm.mlir.addressof @"__aoti_op_name_op_aten::empty_like"
// CHECK-SAME:        : !llvm.ptr
// CHECK:           %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[C6:.*]] = llvm.mlir.constant(6 : i64) : i64
// CHECK:           %[[BOOL_ADAPT:.*]] = builtin.unrealized_conversion_cast
// CHECK-SAME:        : !torch.bool to i1
// CHECK:           %[[SLOTS:.*]] = llvm.alloca %[[C6]] x i64 : (i64) -> !llvm.ptr
// CHECK-NEXT:      %[[V0:.*]] = llvm.ptrtoint %[[ARG0]] : !llvm.ptr to i64
// CHECK-NEXT:      llvm.store %[[V0]], %[[SLOTS]] : i64, !llvm.ptr
// CHECK:           llvm.store %[[C0]]
// CHECK:           llvm.store %[[C0]]
// CHECK:           llvm.store %[[C0]]
// CHECK:           %[[BOOL_EXT:.*]] = llvm.zext %[[BOOL_ADAPT]] : i1 to i64
// CHECK-NEXT:      llvm.store %[[BOOL_EXT]]
// CHECK:           llvm.store %[[C0]]
// CHECK:           %[[RET:.*]] = llvm.call @aoti_torch_call_dispatcher(%[[OPNAME]], %[[OVERLOAD]], %[[SLOTS]])
// CHECK-SAME:        : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:           %[[RES_VAL:.*]] = llvm.load
// CHECK:           %[[RES_PTR:.*]] = llvm.inttoptr
// CHECK-NEXT:      return %[[RES_PTR]] : !llvm.ptr
func.func @torch.aten.empty_like(%arg0: !torch.vtensor<[200,200,26],f64>) -> !torch.vtensor<[200,200,26],f64> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten.empty_like %arg0, %none, %none, %none, %false, %none : !torch.vtensor<[200,200,26],f64>, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[200,200,26],f64>
  return %0 : !torch.vtensor<[200,200,26],f64>
}
