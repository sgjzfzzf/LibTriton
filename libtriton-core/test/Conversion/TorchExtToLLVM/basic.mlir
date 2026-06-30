// RUN: libtriton-core-opt %s --convert-to-llvm -split-input-file | FileCheck %s

// Globals and function declarations at module level (order may vary).
// CHECK-DAG:  llvm.mlir.global internal constant @__libtriton_constant_overload_
// CHECK-DAG:  llvm.mlir.global internal constant @"__libtriton_constant_op_aten::empty_like"
// CHECK-DAG:  llvm.func @aoti_torch_call_dispatcher

// CHECK-LABEL: llvm.func @aten_empty_like(
// CHECK-SAME:    %[[ARG0:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// Allocate slot array (max(num_operands=6, num_results=1) = 6 slots).
// CHECK:         %[[N_SLOTS:.*]] = llvm.mlir.constant(6 : i64) : i64
// CHECK-NEXT:    %[[SLOTS:.*]] = llvm.alloca %[[N_SLOTS]] x i64 : (i64) -> !llvm.ptr
// Convert TVMFFIAny -> extract payload -> inttoptr -> TVMFFIObjectHandle -> AtenTensorHandle via mLibTritonTVMFFIObjectToTensor, then store at slot[0].
// CHECK:         llvm.extractvalue %[[ARG0]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:         llvm.inttoptr {{%.*}} : i64 to !llvm.ptr
// CHECK:         llvm.alloca {{%.*}} x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK:         llvm.call @mLibTritonTVMFFIObjectToTensor({{%.*}},
// CHECK:         %[[ATEN_HDL:.*]] = llvm.load {{%.*}} : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[ARG_I64:.*]] = llvm.ptrtoint %[[ATEN_HDL]] : !llvm.ptr to i64
// CHECK:         llvm.getelementptr %[[SLOTS]][0]
// CHECK:         llvm.store %[[ARG_I64]],
// Store zero (lowered from torch.constant.none / torch.constant.bool false)
// at slots[1..5] for the 5 trailing none/bool arguments of empty_like.
// CHECK:         %[[CZ:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         llvm.getelementptr %[[SLOTS]][1]
// CHECK:         llvm.store {{%.*}},
// CHECK:         llvm.getelementptr %[[SLOTS]][2]
// CHECK:         llvm.store {{%.*}},
// CHECK:         llvm.getelementptr %[[SLOTS]][3]
// CHECK:         llvm.store {{%.*}},
// CHECK:         llvm.getelementptr %[[SLOTS]][4]
// CHECK:         llvm.store {{%.*}},
// CHECK:         llvm.getelementptr %[[SLOTS]][5]
// CHECK:         llvm.store {{%.*}},
// Call dispatcher with (op_name, overload_name, slot_array).
// CHECK:         %[[OPNAME:.*]] = llvm.mlir.addressof @"__libtriton_constant_op_aten::empty_like"
// CHECK:         %[[OVERLOAD:.*]] = llvm.mlir.addressof @__libtriton_constant_overload_
// CHECK:         %[[RET:.*]] = llvm.call @aoti_torch_call_dispatcher(%[[OPNAME]], %[[OVERLOAD]], %[[SLOTS]])
// CHECK-SAME:      : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// Load result from slot[0] and convert i64 -> AtenTensorHandle -> TVMFFIObjectHandle -> TVMFFIAny.
// CHECK:         %[[RES_GEP:.*]] = llvm.getelementptr %[[SLOTS]][0]
// CHECK-NEXT:    %[[RES_VAL:.*]] = llvm.load %[[RES_GEP]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[ATEN_PTR:.*]] = llvm.inttoptr %[[RES_VAL]] : i64 to !llvm.ptr
// CHECK:         llvm.alloca {{%.*}} x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK:         llvm.call @mLibTritonTensorToTVMFFIObject(%[[ATEN_PTR]],
// CHECK:         %[[RES_PTR:.*]] = llvm.load {{%.*}} : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[PTR_I64:.*]] = llvm.ptrtoint %[[RES_PTR]] : !llvm.ptr to i64
// CHECK:         llvm.insertvalue {{%.*}}, {{%.*}}[2] : !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT:    llvm.return {{%.*}} : !llvm.struct<(i32, i32, i64)>

module {
func.func @aten_empty_like(%arg0: !torch.vtensor<[200,200,26],f64>) -> !torch.vtensor<[200,200,26],f64> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten.empty_like %arg0, %none, %none, %none, %false, %none : !torch.vtensor<[200,200,26],f64>, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[200,200,26],f64>
  return %0 : !torch.vtensor<[200,200,26],f64>
}
}

// -----

// CHECK-LABEL: llvm.func @list_construct(
// CHECK-SAME:    %[[A:.*]]: !llvm.struct<(i32, i32, i64)>, %[[B:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK:      %[[N:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[ARR:.*]] = llvm.alloca %[[N]] x !llvm.struct<(i32, i32, i64)>
// CHECK:         llvm.extractvalue %[[A]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:      llvm.getelementptr %[[ARR]][0]
// Fill TVMFFIAny[0] with kTVMFFIInt=1, v_int64=payload(%A)
// CHECK:         llvm.getelementptr {{%.*}}[0, 0]
// CHECK:      llvm.store {{%.*}}, {{%.*}} : i32, !llvm.ptr
// CHECK:      llvm.getelementptr {{%.*}}[0, 1]
// CHECK:      llvm.store {{%.*}}, {{%.*}} : i32, !llvm.ptr
// CHECK:      llvm.getelementptr {{%.*}}[0, 2]
// CHECK:      llvm.store {{%.*}}, {{%.*}} : i64, !llvm.ptr
// CHECK:         llvm.extractvalue %[[B]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:         llvm.getelementptr %[[ARR]][1]
// Fill TVMFFIAny[1] with kTVMFFIInt=1, v_int64=payload(%B)
// CHECK:         llvm.getelementptr {{%.*}}[0, 0]
// CHECK:      llvm.store {{%.*}}, {{%.*}} : i32, !llvm.ptr
// CHECK:      llvm.getelementptr {{%.*}}[0, 1]
// CHECK:      llvm.store {{%.*}}, {{%.*}} : i32, !llvm.ptr
// CHECK:      llvm.getelementptr {{%.*}}[0, 2]
// CHECK:      llvm.store {{%.*}}, {{%.*}} : i64, !llvm.ptr
// Build slot pointers.
// CHECK:      llvm.getelementptr %[[ARR]][0]
// CHECK:      llvm.getelementptr %[[ARR]][1]
// TVMFFIFunctionGetGlobal("ffi.Array", &func)
// CHECK:      llvm.mlir.addressof @__libtriton_constant_ffi.Array_ffi.Array
// CHECK:      llvm.call @TVMFFIFunctionGetGlobal
// CHECK:      llvm.load {{%.*}} : !llvm.ptr -> !llvm.ptr
// Copy slots to contiguous array.
// CHECK:      llvm.load {{%.*}} : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:      llvm.store {{%.*}}, {{%.*}} : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK:      llvm.load {{%.*}} : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:      llvm.store {{%.*}}, {{%.*}} : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// TVMFFIFunctionCall + TVMFFIObjectDecRef
// CHECK:      llvm.call @TVMFFIFunctionCall
// CHECK:      llvm.call @TVMFFIObjectDecRef
// Build TVMFFIAny(kTVMFFIArray=71) from the ffi.Array result, return.
// CHECK:      llvm.getelementptr {{%.*}}[0, 2]
// CHECK:      %[[VOBJ:.*]] = llvm.load {{%.*}} : !llvm.ptr -> i64
// CHECK:      llvm.mlir.undef : !llvm.struct<(i32, i32, i64)>
// CHECK:      llvm.mlir.constant(71 : i32) : i32
// CHECK:      llvm.insertvalue {{%.*}}, {{%.*}}[0] : !llvm.struct<(i32, i32, i64)>
// CHECK:      llvm.insertvalue %[[VOBJ]], {{%.*}}[2] : !llvm.struct<(i32, i32, i64)>
// CHECK:      llvm.return {{%.*}} : !llvm.struct<(i32, i32, i64)>

module {
func.func @list_construct(%arg0: !torch.int, %arg1: !torch.int) -> !torch.list<int> {
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
  return %0 : !torch.list<int>
}
}

// CHECK-LABEL: llvm.func @list_delete_list(
// CHECK-SAME:    %[[LIST:.*]]: !llvm.struct<(i32, i32, i64)>) {
// CHECK-NEXT:    llvm.extractvalue %[[LIST]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:         llvm.inttoptr {{%.*}} : i64 to !llvm.ptr
// CHECK:         llvm.call @TVMFFIObjectDecRef({{%.*}}) : (!llvm.ptr) -> i32
// CHECK-NEXT:    llvm.return

module {
func.func @list_delete_list(%list: !torch.list<int>) {
  torchext.aoti.ListDeleteList %list : !torch.list<int>
  return
}
}

// -----

// Test OptionalHandler: optional tensor input (bias) boxed via malloc.
// Note: function signatures no longer use !torch.optional; the bias is
// passed as a plain !torch.vtensor when present. The c10 schema still
// reports OptionalType, so the Schema dispatch code boxes the value.
// CHECK-DAG:  llvm.mlir.global internal constant @"__libtriton_constant_op_aten::linear"
// CHECK-DAG:  llvm.mlir.global internal constant @__libtriton_constant_overload_
// CHECK-DAG:  llvm.func @aoti_torch_call_dispatcher

// CHECK-LABEL: llvm.func @aten_linear_optional(
// CHECK-SAME:    %[[A0:.*]]: !llvm.struct<(i32, i32, i64)>, %[[A1:.*]]: !llvm.struct<(i32, i32, i64)>, %[[A2:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// Allocate slot array (max(num_operands=3, num_results=1) = 3 slots).
// CHECK:         %[[N_SLOTS:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[SLOTS:.*]] = llvm.alloca %[[N_SLOTS]] x i64 : (i64) -> !llvm.ptr
// Slot 0 (arg0): extractvalue[2] + inttoptr → TVMFFIObjectHandle → AtenTensorHandle.
// CHECK:         llvm.extractvalue %[[A0]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:         llvm.inttoptr {{%.*}} : i64 to !llvm.ptr
// CHECK:         llvm.alloca {{%.*}} x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK:         llvm.call @mLibTritonTVMFFIObjectToTensor({{%.*}},
// CHECK:         %[[A0_ATEN:.*]] = llvm.load {{%.*}} : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[A0_I64:.*]] = llvm.ptrtoint %[[A0_ATEN]] : !llvm.ptr to i64
// CHECK:         llvm.getelementptr %[[SLOTS]][0]
// CHECK:         llvm.store %[[A0_I64]],
// Slot 1 (arg1): extractvalue[2] + inttoptr → TVMFFIObjectHandle → AtenTensorHandle.
// CHECK:         llvm.extractvalue %[[A1]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:         llvm.inttoptr {{%.*}} : i64 to !llvm.ptr
// CHECK:         llvm.alloca {{%.*}} x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK:         llvm.call @mLibTritonTVMFFIObjectToTensor({{%.*}},
// CHECK:         %[[A1_ATEN:.*]] = llvm.load {{%.*}} : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[A1_I64:.*]] = llvm.ptrtoint %[[A1_ATEN]] : !llvm.ptr to i64
// CHECK:         llvm.getelementptr %[[SLOTS]][1]
// CHECK:         llvm.store %[[A1_I64]],
// Slot 2 (bias: optional tensor input → unpack TVMFFIObjectHandle → AtenTensorHandle,
// then box via malloc, flowing through CondBr + merge for the OptionalType encoding).
// The cond_br terminator sits in ^bb0 (before the then/else blocks).
// CHECK:         llvm.cond_br {{%.*}}, ^bb1, ^bb2
// --- None block: zero (None sentinel) ---
// CHECK:         %[[A2_ZERO:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         llvm.br ^bb3(%[[A2_ZERO]] : i64)
// --- Some block: box inner StableIValue on the heap ---
// CHECK:         llvm.extractvalue %[[A2]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:         llvm.inttoptr {{%.*}} : i64 to !llvm.ptr
// CHECK:         llvm.alloca {{%.*}} x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK:         llvm.call @mLibTritonTVMFFIObjectToTensor({{%.*}},
// CHECK:         %[[A2_ATEN:.*]] = llvm.load {{%.*}} : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[A2_I64:.*]] = llvm.ptrtoint %[[A2_ATEN]] : !llvm.ptr to i64
// CHECK:         %[[A2_SIZE:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK:         %[[A2_PTR:.*]] = llvm.call @malloc(%[[A2_SIZE]]) : (i64) -> !llvm.ptr
// CHECK:         llvm.store %[[A2_I64]], %[[A2_PTR]] : i64, !llvm.ptr
// CHECK:         %[[A2_BOXED:.*]] = llvm.ptrtoint %[[A2_PTR]] : !llvm.ptr to i64
// CHECK:         llvm.br ^bb3(%[[A2_BOXED]] : i64)
// CHECK:         llvm.getelementptr %[[SLOTS]][2]
// CHECK:         llvm.store {{%.*}}, {{%.*}} : i64, !llvm.ptr
// Call dispatcher with (op_name, overload_name, slot_array).
// CHECK:         llvm.call @aoti_torch_call_dispatcher
// Load result from slot[0] and convert i64 → !llvm.ptr.
// CHECK:         %[[RES_VAL:.*]] = llvm.load {{%.*}} : !llvm.ptr -> i64
// CHECK-NEXT:    %[[ATEN_PTR:.*]] = llvm.inttoptr %[[RES_VAL]] : i64 to !llvm.ptr
// CHECK:         llvm.alloca {{%.*}} x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK:         llvm.call @mLibTritonTensorToTVMFFIObject(%[[ATEN_PTR]],
// CHECK:         %[[RES_PTR:.*]] = llvm.load {{%.*}} : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[RES_I64:.*]] = llvm.ptrtoint %[[RES_PTR]] : !llvm.ptr to i64
// CHECK:         llvm.insertvalue {{%.*}}, {{%.*}}[2] : !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT:    llvm.return {{%.*}} : !llvm.struct<(i32, i32, i64)>

module {
func.func @aten_linear_optional(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.vtensor<[4,3],f32>, %arg2: !torch.vtensor<[4],f32>) -> !torch.vtensor<[2,4],f32> {
  %0 = torch.aten.linear %arg0, %arg1, %arg2 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[4,3],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}
}
