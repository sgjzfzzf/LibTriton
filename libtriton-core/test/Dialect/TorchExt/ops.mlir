// RUN: libtriton-core-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @object_dec_ref
func.func @object_dec_ref(%object: !torch.list<int>) {
  // CHECK: torchext.ObjectDecRef %[[OBJ:.*]] : !torch.list<int>
  torchext.ObjectDecRef %object : !torch.list<int>
  return
}
