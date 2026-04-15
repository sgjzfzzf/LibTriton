#include <stdlib.h>

#include "dlpack/dlpack.h"

__attribute__((visibility("hidden"))) void *
_mlir_memref_to_llvm_alloc(size_t size) {
  return malloc(size);
}

__attribute__((visibility("hidden"))) void *
_mlir_memref_to_llvm_aligned_alloc(size_t alignment, size_t size) {
  return aligned_alloc(alignment, size);
}

__attribute__((visibility("hidden"))) void
_mlir_memref_to_llvm_free(void *ptr) {
  free(ptr);
}

__attribute__((visibility("hidden"))) void
__libtriton_dlpack_default_managed_tensor_deleter(DLManagedTensor *self) {
  if (self == NULL) {
    return;
  }
  free(self->manager_ctx);
  free(self->dl_tensor.strides);
  free(self->dl_tensor.shape);
  free(self);
}
