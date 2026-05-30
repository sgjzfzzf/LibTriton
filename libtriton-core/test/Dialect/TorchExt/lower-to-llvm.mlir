// RUN: libtriton-core-opt %s -convert-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @kernel(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
        kernel {
      gpu.return
    }
  }
  gpu.module @buf_kernels {
    gpu.func @buf_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
        kernel {
      gpu.return
    }
  }
  gpu.module @async_kernels {
    gpu.func @async_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr,
                           %arg3: i32, %arg4: !llvm.ptr, %arg5: !llvm.ptr)
        kernel {
      gpu.return
    }
  }

  // CHECK-LABEL: llvm.func @lower_scalar_operand
  func.func @lower_scalar_operand(
      %gx: i64, %gy: i64, %gz: i64, %bx: i64, %by: i64, %bz: i64,
      %arg0: i32) {
    torch_ext.triton_kernel_launch @kernels::@kernel
      blocks in(%gx, %gy, %gz) threads in(%bx, %by, %bz) : i64
      args(%arg0 : i32)
    return
    // CHECK-NOT: builtin.unrealized_conversion_cast
    // CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: gpu.launch_func @kernels::@kernel
    // CHECK-SAME: args(%{{.*}} : i32, %[[NULL]] : !llvm.ptr, %[[NULL]] : !llvm.ptr)
  }

  // CHECK-LABEL: llvm.func @lower_memref_operand
  func.func @lower_memref_operand(
      %gx: i64, %gy: i64, %gz: i64, %bx: i64, %by: i64, %bz: i64,
      %buf: memref<?xf32>) {
    torch_ext.triton_kernel_launch @buf_kernels::@buf_kernel
      blocks in(%gx, %gy, %gz) threads in(%bx, %by, %bz) : i64
      args(%buf : memref<?xf32>)
    return
    // CHECK-NOT: builtin.unrealized_conversion_cast
    // CHECK: %[[APTR:.*]] = llvm.extractvalue %{{.*}}[1]
    // CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: gpu.launch_func @buf_kernels::@buf_kernel
    // CHECK-SAME: args(%[[APTR]] : !llvm.ptr, %[[NULL]] : !llvm.ptr, %[[NULL]] : !llvm.ptr)
  }

  // CHECK-LABEL: llvm.func @lower_async_dependency
  func.func @lower_async_dependency(
      %gx: index, %gy: index, %gz: index, %bx: index, %by: index, %bz: index,
      %arg0: memref<4096xf32>, %arg1: memref<4096xf32>) {
    %c0_i32 = arith.constant 0 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %stream = torch_ext.get_current_stream device = -1 : !gpu.async.token
    %memref = gpu.alloc  () : memref<4096xf32>
    %launch = torch_ext.triton_kernel_launch async [%stream] @async_kernels::@async_kernel
        blocks in(%gx, %gy, %gz) threads in(%bx, %by, %bz)
        dynamic_shared_memory_size %c0_i32
        args(%arg0, %arg1, %memref, %c4096_i32 : memref<4096xf32>, memref<4096xf32>, memref<4096xf32>, i32)
    return
    // CHECK: llvm.call @__libtriton_get_current_stream
    // CHECK: gpu.launch_func async [%{{.*}}] @async_kernels::@async_kernel blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}}) dynamic_shared_memory_size %{{.*}} args(%{{.*}} : !llvm.ptr, %{{.*}} : !llvm.ptr, %{{.*}} : !llvm.ptr, %{{.*}} : i32, %{{.*}} : !llvm.ptr, %{{.*}} : !llvm.ptr)
  }
}

// CHECK-LABEL: llvm.func @lowering_get_current_device
// CHECK-SAME: () -> !llvm.struct<packed (i32, i32)>
func.func @lowering_get_current_device() -> !dlpack.device {
  // CHECK-NOT: torch_ext.get_current_device
  // CHECK: %[[DEV:.*]] = llvm.call @__libtriton_get_current_device() : () -> !llvm.struct<packed (i32, i32)>
  // CHECK: return %[[DEV]] : !llvm.struct<packed (i32, i32)>
  %0 = torch_ext.get_current_device : !dlpack.device
  return %0 : !dlpack.device
}

// CHECK-LABEL: llvm.func @lowering_get_current_stream_with_device
func.func @lowering_get_current_stream_with_device() {
  // CHECK-NOT: torch_ext.get_current_stream
  // CHECK: %[[DEVICE:.*]] = llvm.mlir.constant(7 : i8) : i8
  // CHECK: llvm.call @__libtriton_get_current_stream(%[[DEVICE]]) : (i8) -> !llvm.ptr
  %0 = torch_ext.get_current_stream device = 7 : !gpu.async.token
  return
}

// CHECK-LABEL: llvm.func @lowering_get_current_stream_default_device
func.func @lowering_get_current_stream_default_device() {
  // CHECK-NOT: torch_ext.get_current_stream
  // CHECK: %[[DEFAULT_DEVICE:.*]] = llvm.mlir.constant(-1 : i8) : i8
  // CHECK: llvm.call @__libtriton_get_current_stream(%[[DEFAULT_DEVICE]]) : (i8) -> !llvm.ptr
  %0 = torch_ext.get_current_stream device = -1 : !gpu.async.token
  return
}
