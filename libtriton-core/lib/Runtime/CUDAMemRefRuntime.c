#include <cuda_runtime.h>
#include <stdlib.h>

#include "dlpack/dlpack.h"

__attribute__((visibility("hidden"))) void *
_mlir_memref_to_llvm_alloc(size_t size) {
  void *ptr = NULL;
  cudaError_t err = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
  return (err == cudaSuccess) ? ptr : NULL;
}

__attribute__((visibility("hidden"))) void *
_mlir_memref_to_llvm_aligned_alloc(size_t alignment, size_t size) {
  // cudaMallocManaged guarantees 256-byte alignment; ignore alignment hint.
  (void)alignment;
  return _mlir_memref_to_llvm_alloc(size);
}

__attribute__((visibility("hidden"))) void
_mlir_memref_to_llvm_free(void *ptr) {
  cudaFree(ptr);
}

__attribute__((visibility("hidden"))) void
__libtriton_dlpack_default_managed_tensor_deleter(DLManagedTensor *self) {
  if (self == NULL) {
    return;
  }
  cudaFree(self->manager_ctx);
  free(self->dl_tensor.strides);
  free(self->dl_tensor.shape);
  free(self);
}
