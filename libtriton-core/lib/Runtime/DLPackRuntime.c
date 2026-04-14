#include <stdlib.h>

#include "dlpack/dlpack.h"

__attribute__((visibility("default"))) void
__libtriton_dlpack_default_managed_tensor_deleter(void *self) {
  DLManagedTensor *managedTensor;

  if (self == NULL) {
    return;
  }

  managedTensor = (DLManagedTensor *)self;
  free(managedTensor->dl_tensor.data);
  free(managedTensor->dl_tensor.strides);
  free(managedTensor->dl_tensor.shape);
  free(managedTensor);
}
