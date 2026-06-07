// RUN: libtriton-core-opt %s -split-input-file -convert-tvm-ffi-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @__tvm_ffi_void_func(
// CHECK-SAME:    %{{.*}}: !llvm.ptr, %[[ARGS_PTR:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK-NEXT:    %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return %[[ZERO]] : i32
tvm_ffi.func @void_func() {
  tvm_ffi.return
}

// -----

// CHECK-LABEL: llvm.func @__tvm_ffi_make_int(
// CHECK-SAME:    %{{.*}}: !llvm.ptr, %[[ARGS_PTR:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK:         %[[C42:.*]] = torch.constant.int 42
// CHECK-NEXT:    %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[C42]] : !torch.int to i64
// CHECK-NEXT:    %[[PAYLOAD:.*]] = llvm.getelementptr %[[RET_PTR]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    llvm.store %[[CAST]], %[[PAYLOAD]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return %[[ZERO]] : i32
tvm_ffi.func @make_int() -> !torch.int {
  %0 = torch.constant.int 42
  tvm_ffi.return %0 : !torch.int
}

// -----

// CHECK-LABEL: llvm.func @__tvm_ffi_print_int(
// CHECK-SAME:    %{{.*}}: !llvm.ptr, %[[ARGS_PTR:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK:         %[[SLOT:.*]] = llvm.getelementptr %[[ARGS_PTR]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    %[[PAY:.*]] = llvm.getelementptr %[[SLOT]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    %[[RAW:.*]] = llvm.load %[[PAY]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[RAW]] : i64 to !torch.int
// CHECK-NEXT:    %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return %[[ZERO]] : i32
tvm_ffi.func @print_int(%arg0: !torch.int) {
  tvm_ffi.return
}

// -----

// CHECK-LABEL: llvm.func @__tvm_ffi_identity(
// CHECK-SAME:    %{{.*}}: !llvm.ptr, %[[ARGS_PTR:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK:         %[[SLOT:.*]] = llvm.getelementptr %[[ARGS_PTR]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[PAY:.*]] = llvm.getelementptr %[[SLOT]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[RAW:.*]] = llvm.load %[[PAY]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[TO_TORCH:.*]] = builtin.unrealized_conversion_cast %[[RAW]] : i64 to !torch.int
// CHECK-NEXT:    %[[TO_LLVM:.*]] = builtin.unrealized_conversion_cast %[[TO_TORCH]] : !torch.int to i64
// CHECK-NEXT:    %[[RET_PAY:.*]] = llvm.getelementptr %[[RET_PTR]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    llvm.store %[[TO_LLVM]], %[[RET_PAY]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return %[[ZERO]] : i32
tvm_ffi.func @identity(%arg0: !torch.int) -> !torch.int {
  tvm_ffi.return %arg0 : !torch.int
}

// -----

// CHECK-LABEL: llvm.func @__tvm_ffi_identity_bool(
// CHECK-SAME:    %{{.*}}: !llvm.ptr, %[[ARGS_PTR:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK:         %[[SLOT:.*]] = llvm.getelementptr %[[ARGS_PTR]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[PAY:.*]] = llvm.getelementptr %[[SLOT]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[RAW:.*]] = llvm.load %[[PAY]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[TRUNC:.*]] = llvm.trunc %[[RAW]] : i64 to i1
// CHECK-NEXT:    %[[TO_TORCH:.*]] = builtin.unrealized_conversion_cast %[[TRUNC]] : i1 to !torch.bool
// CHECK-NEXT:    %[[TO_I1:.*]] = builtin.unrealized_conversion_cast %[[TO_TORCH]] : !torch.bool to i1
// CHECK:         %[[RET_PAY:.*]] = llvm.getelementptr %[[RET_PTR]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    %[[ZEXT:.*]] = llvm.zext %[[TO_I1]] : i1 to i64
// CHECK-NEXT:    llvm.store %[[ZEXT]], %[[RET_PAY]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return %[[ZERO]] : i32
tvm_ffi.func @identity_bool(%arg0: !torch.bool) -> !torch.bool {
  tvm_ffi.return %arg0 : !torch.bool
}

// -----

// CHECK-LABEL: llvm.func @__tvm_ffi_identity_float(
// CHECK-SAME:    %{{.*}}: !llvm.ptr, %[[ARGS_PTR:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK:         %[[SLOT:.*]] = llvm.getelementptr %[[ARGS_PTR]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[PAY:.*]] = llvm.getelementptr %[[SLOT]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[RAW:.*]] = llvm.load %[[PAY]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[TO_F64:.*]] = llvm.bitcast %[[RAW]] : i64 to f64
// CHECK-NEXT:    %[[TO_TORCH:.*]] = builtin.unrealized_conversion_cast %[[TO_F64]] : f64 to !torch.float
// CHECK-NEXT:    %[[TO_F64_2:.*]] = builtin.unrealized_conversion_cast %[[TO_TORCH]] : !torch.float to f64
// CHECK:         %[[RET_PAY:.*]] = llvm.getelementptr %[[RET_PTR]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    %[[TO_I64:.*]] = llvm.bitcast %[[TO_F64_2]] : f64 to i64
// CHECK-NEXT:    llvm.store %[[TO_I64]], %[[RET_PAY]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return %[[ZERO]] : i32
tvm_ffi.func @identity_float(%arg0: !torch.float) -> !torch.float {
  tvm_ffi.return %arg0 : !torch.float
}

// -----

// CHECK-LABEL: llvm.func @__tvm_ffi_add(
// CHECK-SAME:    %{{.*}}: !llvm.ptr, %[[ARGS_PTR:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK:         %[[SLOT0:.*]] = llvm.getelementptr %[[ARGS_PTR]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[PAY0:.*]] = llvm.getelementptr %[[SLOT0]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[RAW0:.*]] = llvm.load %[[PAY0]] : !llvm.ptr -> i64
// CHECK:         %[[ARG0:.*]] = builtin.unrealized_conversion_cast %[[RAW0]] : i64 to !torch.int
// CHECK:         %[[SLOT1:.*]] = llvm.getelementptr %[[ARGS_PTR]][1]
// CHECK:         %[[PAY1:.*]] = llvm.getelementptr %[[SLOT1]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[RAW1:.*]] = llvm.load %[[PAY1]] : !llvm.ptr -> i64
// CHECK:         %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[RAW1]] : i64 to !torch.int
// CHECK:         %[[ADD:.*]] = torch.aten.add %[[ARG0]], %[[ARG1]] : !torch.int, !torch.int -> !torch.int
// CHECK:         %[[TO_I64:.*]] = builtin.unrealized_conversion_cast %[[ADD]] : !torch.int to i64
// CHECK-NEXT:    %[[RET_PAY:.*]] = llvm.getelementptr %[[RET_PTR]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    llvm.store %[[TO_I64]], %[[RET_PAY]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return %[[ZERO]] : i32
tvm_ffi.func @add(%arg0: !torch.int, %arg1: !torch.int) -> !torch.int {
  %0 = torch.aten.add %arg0, %arg1 : !torch.int, !torch.int -> !torch.int
  tvm_ffi.return %0 : !torch.int
}

// -----

// Multi-block: a conditional branch using torch.prim.If.
// CHECK-LABEL: llvm.func @__tvm_ffi_cond_identity(
// CHECK-SAME:    %{{.*}}: !llvm.ptr, %[[ARGS_PTR:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK:         %[[SLOT:.*]] = llvm.getelementptr %[[ARGS_PTR]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[PAY:.*]] = llvm.getelementptr %[[SLOT]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[RAW:.*]] = llvm.load %[[PAY]] : !llvm.ptr -> i64
// CHECK:         %[[ARG:.*]] = builtin.unrealized_conversion_cast %[[RAW]] : i64 to !torch.int
// CHECK:         %[[GT:.*]] = torch.aten.gt.int %[[ARG]], %[[ARG]] : !torch.int, !torch.int -> !torch.bool
// CHECK:         %[[RESULT:.*]] = torch.prim.If %[[GT]] -> (!torch.int) {
// CHECK:         }
// CHECK:         %[[TO_I64:.*]] = builtin.unrealized_conversion_cast %[[RESULT]] : !torch.int to i64
// CHECK:         %[[RET_PAY:.*]] = llvm.getelementptr %[[RET_PTR]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         llvm.store %[[TO_I64]], %[[RET_PAY]] : i64, !llvm.ptr
// CHECK:         %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         llvm.return %[[ZERO]] : i32
tvm_ffi.func @cond_identity(%arg0: !torch.int) -> !torch.int {
  %cond = torch.aten.gt.int %arg0, %arg0 : !torch.int, !torch.int -> !torch.bool
  %result = torch.prim.If %cond -> (!torch.int) {
    torch.prim.If.yield %arg0 : !torch.int
  } else {
    %zero = torch.constant.int 0
    torch.prim.If.yield %zero : !torch.int
  }
  tvm_ffi.return %result : !torch.int
}

// -----

// Tensor input: load DLTensor from TVMFFIAny and convert to AtenTensorHandle.
// CHECK-LABEL: llvm.func @__tvm_ffi_tensor_func(
// CHECK-SAME:    %{{.*}}: !llvm.ptr, %[[ARGS_PTR:.*]]: !llvm.ptr,
// CHECK-SAME:    %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK:         %[[SLOT:.*]] = llvm.getelementptr %[[ARGS_PTR]][0]
// CHECK-NEXT:    %[[TYPEIDX_PTR:.*]] = llvm.getelementptr %[[SLOT]][0]
// CHECK-NEXT:    %[[TYPE_INDEX:.*]] = llvm.load %[[TYPEIDX_PTR]] : !llvm.ptr -> i32
// CHECK-NEXT:    %[[PAYLOAD_PTR:.*]] = llvm.getelementptr %[[SLOT]]{{\[}}0, 2]
// CHECK-NEXT:    %[[RAW_PAYLOAD:.*]] = llvm.load %[[PAYLOAD_PTR]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[RAW_PTR:.*]] = llvm.inttoptr %[[RAW_PAYLOAD]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[OBJ_THRESH:.*]] = llvm.mlir.constant(70 : i32) : i32
// CHECK-NEXT:    %[[IS_TENSOR_OBJ:.*]] = llvm.icmp "eq" %[[TYPE_INDEX]], %[[OBJ_THRESH]] : i32
// CHECK-NEXT:    %[[RAW_I64:.*]] = llvm.ptrtoint %[[RAW_PTR]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[OBJ_OFFSET:.*]] = llvm.mlir.constant(24 : i64) : i64
// CHECK-NEXT:    %[[OFFSET_I64:.*]] = llvm.add %[[RAW_I64]], %[[OBJ_OFFSET]] : i64
// CHECK-NEXT:    %[[OFFSET_PTR:.*]] = llvm.inttoptr %[[OFFSET_I64]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[DLTENSOR:.*]] = llvm.select %[[IS_TENSOR_OBJ]], %[[OFFSET_PTR]], %[[RAW_PTR]] : i1, !llvm.ptr
// CHECK-NEXT:    %[[DATA_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 0]
// CHECK-NEXT:    %[[DATA:.*]] = llvm.load %[[DATA_PTR]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %[[DEVTYPE_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 1, 0]
// CHECK-NEXT:    %[[DL_DEV_TYPE:.*]] = llvm.load %[[DEVTYPE_PTR]] : !llvm.ptr -> i32
// CHECK-NEXT:    %[[DEVID_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 1, 1]
// CHECK-NEXT:    %[[DEVICE_ID:.*]] = llvm.load %[[DEVID_PTR]] : !llvm.ptr -> i32
// CHECK-NEXT:    %[[NDIM_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 2]
// CHECK-NEXT:    %[[NDIM:.*]] = llvm.load %[[NDIM_PTR]] : !llvm.ptr -> i32
// CHECK-NEXT:    %[[DTCODE_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 3, 0]
// CHECK-NEXT:    %[[DTYPE_CODE:.*]] = llvm.load %[[DTCODE_PTR]] : !llvm.ptr -> i8
// CHECK-NEXT:    %[[DTBITS_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 3, 1]
// CHECK-NEXT:    %[[DTYPE_BITS:.*]] = llvm.load %[[DTBITS_PTR]] : !llvm.ptr -> i8
// CHECK-NEXT:    %[[SHAPE_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 4]
// CHECK-NEXT:    %[[SHAPE:.*]] = llvm.load %[[SHAPE_PTR]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %[[STRIDES_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 5]
// CHECK-NEXT:    %[[STRIDES:.*]] = llvm.load %[[STRIDES_PTR]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %[[BOFF_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 6]
// CHECK-NEXT:    %[[BYTE_OFF:.*]] = llvm.load %[[BOFF_PTR]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[TORCH_DTYPE:.*]] = llvm.call @mLibTritonTVMFFIToTorchType(%[[DTYPE_CODE]], %[[DTYPE_BITS]]) : (i8, i8) -> i32
// CHECK-NEXT:    %[[DEVICE_TYPE:.*]] = llvm.call @mLibTritonTVMFFIDeviceToTorchDeviceType(%[[DL_DEV_TYPE]]) : (i32) -> i32
// CHECK-NEXT:    %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[HANDLE_SLOT:.*]] = llvm.alloca %[[ONE]] x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[NDIM_I64:.*]] = llvm.sext %[[NDIM]] : i32 to i64
// CHECK-NEXT:    %[[RET:.*]] = llvm.call @aoti_torch_create_tensor_from_blob(%[[DATA]], %[[NDIM_I64]], %[[SHAPE]], %[[STRIDES]], %[[BYTE_OFF]], %[[TORCH_DTYPE]], %[[DEVICE_TYPE]], %[[DEVICE_ID]], %[[HANDLE_SLOT]]) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:    %[[HANDLE:.*]] = llvm.load %[[HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[HANDLE]] : !llvm.ptr to !torch.tensor
// CHECK-NEXT:    %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return %[[ZERO]] : i32
tvm_ffi.func @tensor_func(%arg0: !torch.tensor) {
  tvm_ffi.return
}

// -----

// Tensor identity: loads tensor from input TVMFFIAny slot, stores it back to
// the output TVMFFIAny slot. Exercises both BaseTensorHandler::load and ::store.
// CHECK-LABEL: llvm.func @__tvm_ffi_tensor_identity(
// CHECK-SAME:    %[[HANDLE:.*]]: !llvm.ptr, %[[ARGS_PTR:.*]]: !llvm.ptr,
// CHECK-SAME:    %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
//
// --- Load path: extract DLTensor from input TVMFFIAny ---
// CHECK:         %[[SLOT:.*]] = llvm.getelementptr %[[ARGS_PTR]][0]
// CHECK-NEXT:    %[[TYPEIDX_PTR:.*]] = llvm.getelementptr %[[SLOT]][0]
// CHECK-NEXT:    %[[TYPE_INDEX:.*]] = llvm.load %[[TYPEIDX_PTR]] : !llvm.ptr -> i32
// CHECK-NEXT:    %[[PAYLOAD_PTR:.*]] = llvm.getelementptr %[[SLOT]]{{\[}}0, 2]
// CHECK-NEXT:    %[[RAW_PAYLOAD:.*]] = llvm.load %[[PAYLOAD_PTR]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[RAW_PTR:.*]] = llvm.inttoptr %[[RAW_PAYLOAD]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[OBJ_THRESH:.*]] = llvm.mlir.constant(70 : i32) : i32
// CHECK-NEXT:    %[[IS_TENSOR_OBJ:.*]] = llvm.icmp "eq" %[[TYPE_INDEX]], %[[OBJ_THRESH]] : i32
// CHECK-NEXT:    %[[RAW_I64:.*]] = llvm.ptrtoint %[[RAW_PTR]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[OBJ_OFFSET:.*]] = llvm.mlir.constant(24 : i64) : i64
// CHECK-NEXT:    %[[OFFSET_I64:.*]] = llvm.add %[[RAW_I64]], %[[OBJ_OFFSET]] : i64
// CHECK-NEXT:    %[[OFFSET_PTR:.*]] = llvm.inttoptr %[[OFFSET_I64]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[DLTENSOR:.*]] = llvm.select %[[IS_TENSOR_OBJ]], %[[OFFSET_PTR]], %[[RAW_PTR]] : i1, !llvm.ptr
// CHECK-NEXT:    %[[DATA_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 0]
// CHECK-NEXT:    %[[DATA:.*]] = llvm.load %[[DATA_PTR]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %[[DEVTYPE_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 1, 0]
// CHECK-NEXT:    %[[DL_DEV_TYPE:.*]] = llvm.load %[[DEVTYPE_PTR]] : !llvm.ptr -> i32
// CHECK-NEXT:    %[[DEVID_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 1, 1]
// CHECK-NEXT:    %[[DEVICE_ID:.*]] = llvm.load %[[DEVID_PTR]] : !llvm.ptr -> i32
// CHECK-NEXT:    %[[NDIM_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 2]
// CHECK-NEXT:    %[[NDIM:.*]] = llvm.load %[[NDIM_PTR]] : !llvm.ptr -> i32
// CHECK-NEXT:    %[[DTCODE_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 3, 0]
// CHECK-NEXT:    %[[DTYPE_CODE:.*]] = llvm.load %[[DTCODE_PTR]] : !llvm.ptr -> i8
// CHECK-NEXT:    %[[DTBITS_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 3, 1]
// CHECK-NEXT:    %[[DTYPE_BITS:.*]] = llvm.load %[[DTBITS_PTR]] : !llvm.ptr -> i8
// CHECK-NEXT:    %[[SHAPE_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 4]
// CHECK-NEXT:    %[[SHAPE:.*]] = llvm.load %[[SHAPE_PTR]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %[[STRIDES_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 5]
// CHECK-NEXT:    %[[STRIDES:.*]] = llvm.load %[[STRIDES_PTR]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %[[BOFF_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 6]
// CHECK-NEXT:    %[[BYTE_OFF:.*]] = llvm.load %[[BOFF_PTR]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[TORCH_DTYPE:.*]] = llvm.call @mLibTritonTVMFFIToTorchType(%[[DTYPE_CODE]], %[[DTYPE_BITS]]) : (i8, i8) -> i32
// CHECK-NEXT:    %[[DEVICE_TYPE:.*]] = llvm.call @mLibTritonTVMFFIDeviceToTorchDeviceType(%[[DL_DEV_TYPE]]) : (i32) -> i32
// CHECK-NEXT:    %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[HANDLE_SLOT:.*]] = llvm.alloca %[[ONE]] x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[NDIM_I64:.*]] = llvm.sext %[[NDIM]] : i32 to i64
// CHECK-NEXT:    %[[CREATE_RET:.*]] = llvm.call @aoti_torch_create_tensor_from_blob(%[[DATA]], %[[NDIM_I64]], %[[SHAPE]], %[[STRIDES]], %[[BYTE_OFF]], %[[TORCH_DTYPE]], %[[DEVICE_TYPE]], %[[DEVICE_ID]], %[[HANDLE_SLOT]]) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:    %[[TENSOR_HANDLE:.*]] = llvm.load %[[HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %[[TO_TORCH:.*]] = builtin.unrealized_conversion_cast %[[TENSOR_HANDLE]] : !llvm.ptr to !torch.tensor
// CHECK-NEXT:    %[[TO_LLVM:.*]] = builtin.unrealized_conversion_cast %[[TO_TORCH]] : !torch.tensor to !llvm.ptr
//
// --- Store path: extract tensor properties via aoti_torch_get_* ---
// CHECK-NEXT:    %[[ONE2:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[DIM_SLOT:.*]] = llvm.alloca %[[ONE2]] x i64 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[GETDIM_RET:.*]] = llvm.call @aoti_torch_get_dim(%[[TO_LLVM]], %[[DIM_SLOT]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:    %[[NDIM_O:.*]] = llvm.load %[[DIM_SLOT]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[ONE3:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[SIZES_SLOT:.*]] = llvm.alloca %[[ONE3]] x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[GETSIZES_RET:.*]] = llvm.call @aoti_torch_get_sizes(%[[TO_LLVM]], %[[SIZES_SLOT]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:    %[[SIZES_O:.*]] = llvm.load %[[SIZES_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %[[ONE4:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[STRIDES_SLOT:.*]] = llvm.alloca %[[ONE4]] x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[GETSTRIDES_RET:.*]] = llvm.call @aoti_torch_get_strides(%[[TO_LLVM]], %[[STRIDES_SLOT]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:    %[[STRIDES_O:.*]] = llvm.load %[[STRIDES_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %[[ONE5:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[DATA_SLOT:.*]] = llvm.alloca %[[ONE5]] x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[GETDATA_RET:.*]] = llvm.call @aoti_torch_get_data_ptr(%[[TO_LLVM]], %[[DATA_SLOT]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:    %[[DATA_O:.*]] = llvm.load %[[DATA_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %[[ONE6:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[DTYPE_SLOT:.*]] = llvm.alloca %[[ONE6]] x i32 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[GETDTYPE_RET:.*]] = llvm.call @aoti_torch_get_dtype(%[[TO_LLVM]], %[[DTYPE_SLOT]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:    %[[TORCH_DTYPE_O:.*]] = llvm.load %[[DTYPE_SLOT]] : !llvm.ptr -> i32
// CHECK-NEXT:    %[[ONE7:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[DEVTYPE_SLOT:.*]] = llvm.alloca %[[ONE7]] x i32 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[GETDEVTYPE_RET:.*]] = llvm.call @aoti_torch_get_device_type(%[[TO_LLVM]], %[[DEVTYPE_SLOT]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:    %[[TORCH_DEV_TYPE_O:.*]] = llvm.load %[[DEVTYPE_SLOT]] : !llvm.ptr -> i32
// CHECK-NEXT:    %[[ONE8:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[DEVIDX_SLOT:.*]] = llvm.alloca %[[ONE8]] x i32 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[GETDEVIDX_RET:.*]] = llvm.call @aoti_torch_get_device_index(%[[TO_LLVM]], %[[DEVIDX_SLOT]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:    %[[DEV_IDX_O:.*]] = llvm.load %[[DEVIDX_SLOT]] : !llvm.ptr -> i32
// CHECK-NEXT:    %[[ONE9:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[BOFF_SLOT:.*]] = llvm.alloca %[[ONE9]] x i64 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[GETBOFF_RET:.*]] = llvm.call @aoti_torch_get_storage_offset(%[[TO_LLVM]], %[[BOFF_SLOT]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:    %[[BYTE_OFF_O:.*]] = llvm.load %[[BOFF_SLOT]] : !llvm.ptr -> i64
//
// --- Reverse dtype/device mapping (runtime helpers) ---
// CHECK-NEXT:    %[[DLDATATYPE:.*]] = llvm.call @mLibTritonTorchToTVMFFIDtype(%[[TORCH_DTYPE_O]]) : (i32) -> !llvm.struct<packed (i8, i8, i16)>
// CHECK-NEXT:    %[[DLPACK_CODE:.*]] = llvm.extractvalue %[[DLDATATYPE]][0] : !llvm.struct<packed (i8, i8, i16)>
// CHECK-NEXT:    %[[DLPACK_BITS:.*]] = llvm.extractvalue %[[DLDATATYPE]][1] : !llvm.struct<packed (i8, i8, i16)>
// CHECK-NEXT:    %[[DLPACK_LANES:.*]] = llvm.extractvalue %[[DLDATATYPE]][2] : !llvm.struct<packed (i8, i8, i16)>
// CHECK-NEXT:    %[[DLPACK_DEVICE:.*]] = llvm.call @mLibTritonTorchToTVMFFIDevice(%[[TORCH_DEV_TYPE_O]]) : (i32) -> i32
//
// --- Allocate DLManagedTensor via malloc ---
// CHECK-NEXT:    %[[MALLOC_SIZE:.*]] = llvm.mlir.constant(64 : i64) : i64
// CHECK-NEXT:    %[[DLMAN_PTR:.*]] = llvm.call @malloc(%[[MALLOC_SIZE]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[NDIM_I32:.*]] = llvm.trunc %[[NDIM_O]] : i64 to i32
//
// --- Fill DLTensor fields ---
// CHECK-NEXT:    %[[FIELD_DATA:.*]] = llvm.getelementptr %[[DLMAN_PTR]][0, 0, 0]
// CHECK-NEXT:    llvm.store %[[DATA_O]], %[[FIELD_DATA]] : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[FIELD_DEV_TYPE:.*]] = llvm.getelementptr %[[DLMAN_PTR]][0, 0, 1, 0]
// CHECK-NEXT:    llvm.store %[[DLPACK_DEVICE]], %[[FIELD_DEV_TYPE]] : i32, !llvm.ptr
// CHECK-NEXT:    %[[FIELD_DEV_ID:.*]] = llvm.getelementptr %[[DLMAN_PTR]][0, 0, 1, 1]
// CHECK-NEXT:    llvm.store %[[DEV_IDX_O]], %[[FIELD_DEV_ID]] : i32, !llvm.ptr
// CHECK-NEXT:    %[[FIELD_NDIM:.*]] = llvm.getelementptr %[[DLMAN_PTR]][0, 0, 2]
// CHECK-NEXT:    llvm.store %[[NDIM_I32]], %[[FIELD_NDIM]] : i32, !llvm.ptr
// CHECK-NEXT:    %[[FIELD_DTCODE:.*]] = llvm.getelementptr %[[DLMAN_PTR]][0, 0, 3, 0]
// CHECK-NEXT:    llvm.store %[[DLPACK_CODE]], %[[FIELD_DTCODE]] : i8, !llvm.ptr
// CHECK-NEXT:    %[[FIELD_DTBITS:.*]] = llvm.getelementptr %[[DLMAN_PTR]][0, 0, 3, 1]
// CHECK-NEXT:    llvm.store %[[DLPACK_BITS]], %[[FIELD_DTBITS]] : i8, !llvm.ptr
// CHECK-NEXT:    %[[FIELD_LANES:.*]] = llvm.getelementptr %[[DLMAN_PTR]][0, 0, 3, 2]
// CHECK-NEXT:    llvm.store %[[DLPACK_LANES]], %[[FIELD_LANES]] : i16, !llvm.ptr
// CHECK-NEXT:    %[[FIELD_SHAPE:.*]] = llvm.getelementptr %[[DLMAN_PTR]][0, 0, 4]
// CHECK-NEXT:    llvm.store %[[SIZES_O]], %[[FIELD_SHAPE]] : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[FIELD_STRIDES:.*]] = llvm.getelementptr %[[DLMAN_PTR]][0, 0, 5]
// CHECK-NEXT:    llvm.store %[[STRIDES_O]], %[[FIELD_STRIDES]] : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[FIELD_BOFF:.*]] = llvm.getelementptr %[[DLMAN_PTR]][0, 0, 6]
// CHECK-NEXT:    llvm.store %[[BYTE_OFF_O]], %[[FIELD_BOFF]] : i64, !llvm.ptr
//
// --- Set manager_ctx and deleter ---
// CHECK-NEXT:    %[[FIELD_CTX:.*]] = llvm.getelementptr %[[DLMAN_PTR]][0, 1]
// CHECK-NEXT:    llvm.store %[[TO_LLVM]], %[[FIELD_CTX]] : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[DELETER_ADDR:.*]] = llvm.mlir.addressof @mLibTritonDLManagedTensorDeleter : !llvm.ptr
// CHECK-NEXT:    %[[FIELD_DELETER:.*]] = llvm.getelementptr %[[DLMAN_PTR]][0, 2]
// CHECK-NEXT:    llvm.store %[[DELETER_ADDR]], %[[FIELD_DELETER]] : !llvm.ptr, !llvm.ptr
//
// --- Call TVMFFITensorFromDLPack ---
// CHECK-NEXT:    %[[OBJ_SLOT_SIZE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[OBJ_SLOT:.*]] = llvm.alloca %[[OBJ_SLOT_SIZE]] x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[ZERO_I32_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[ZERO_I32_2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[TVM_RET:.*]] = llvm.call @TVMFFITensorFromDLPack(%[[DLMAN_PTR]], %[[ZERO_I32_1]], %[[ZERO_I32_2]], %[[OBJ_SLOT]]) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:    %[[OBJ_HANDLE:.*]] = llvm.load %[[OBJ_SLOT]] : !llvm.ptr -> !llvm.ptr
//
// --- Store result into output TVMFFIAny ---
// CHECK-NEXT:    %[[RET_TYPEIDX:.*]] = llvm.getelementptr %[[RET_PTR]][0]
// CHECK-NEXT:    %[[TENSOR_INDEX:.*]] = llvm.mlir.constant(70 : i32) : i32
// CHECK-NEXT:    llvm.store %[[TENSOR_INDEX]], %[[RET_TYPEIDX]] : i32, !llvm.ptr
// CHECK-NEXT:    %[[RET_PAYLOAD:.*]] = llvm.getelementptr %[[RET_PTR]]{{\[}}0, 2]
// CHECK-NEXT:    %[[HANDLE_I64:.*]] = llvm.ptrtoint %[[OBJ_HANDLE]] : !llvm.ptr to i64
// CHECK-NEXT:    llvm.store %[[HANDLE_I64]], %[[RET_PAYLOAD]] : i64, !llvm.ptr
//
// CHECK-NEXT:    %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return %[[ZERO]] : i32
tvm_ffi.func @tensor_identity(%arg0: !torch.tensor) -> !torch.tensor {
  tvm_ffi.return %arg0 : !torch.tensor
}
