// RUN: libtriton-core-opt -torchext-async-kernel-launch %s | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @async_launch_kernel() kernel {
      gpu.return
    }
  }

  func.func @async_torch_ext_kernel_launch(%gx: index, %gy: index, %gz: index,
                                           %bx: index, %by: index, %bz: index) {
    torch_ext.triton_kernel_launch @async_launch_kernel blocks in(%gx, %gy, %gz) threads in(%bx, %by, %bz)
    return
  }

  // CHECK-LABEL: func.func @async_torch_ext_kernel_launch
  // CHECK: %[[S0:.*]] = torch_ext.get_current_stream device = -1 : !gpu.async.token
  // CHECK: %[[T0:.*]] = torch_ext.triton_kernel_launch async [%[[S0]]] @async_launch_kernel blocks in(%{{.*}}, %{{.*}}, %{{.*}}) threads in(%{{.*}}, %{{.*}}, %{{.*}})

  // -----

  func.func @async_gpu_launch_func(%gx: index, %gy: index, %gz: index,
                                   %bx: index, %by: index, %bz: index) {
    gpu.launch_func @kernels::@async_launch_kernel blocks in (%gx, %gy, %gz) threads in (%bx, %by, %bz)
    return
  }

  // CHECK-LABEL: func.func @async_gpu_launch_func
  // CHECK: gpu.launch_func @kernels::@async_launch_kernel blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}})

  // -----

  func.func @keep_existing_async_stream(%gx: index, %gy: index, %gz: index,
                                        %bx: index, %by: index, %bz: index) {
    %stream = torch_ext.get_current_stream device = -1 : !gpu.async.token
    %triton_result = torch_ext.triton_kernel_launch async [%stream] @async_launch_kernel blocks in(%gx, %gy, %gz) threads in(%bx, %by, %bz)
    %gpu_result = gpu.launch_func async [%stream] @kernels::@async_launch_kernel blocks in (%gx, %gy, %gz) threads in (%bx, %by, %bz)
    return
  }

  // CHECK-LABEL: func.func @keep_existing_async_stream
  // CHECK: %[[S2:.*]] = torch_ext.get_current_stream device = -1 : !gpu.async.token
  // CHECK: torch_ext.triton_kernel_launch async [%[[S2]]] @async_launch_kernel blocks in(%{{.*}}, %{{.*}}, %{{.*}}) threads in(%{{.*}}, %{{.*}}, %{{.*}})
  // CHECK: gpu.launch_func async
  // CHECK-SAME: @kernels::@async_launch_kernel blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}})
}