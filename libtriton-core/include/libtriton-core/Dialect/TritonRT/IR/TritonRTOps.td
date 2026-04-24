#ifndef LIBTRITON_CORE_DIALECT_TRITONRT_IR_TRITONRTOPS_TD_
#define LIBTRITON_CORE_DIALECT_TRITONRT_IR_TRITONRTOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "torch-mlir/Dialect/Torch/IR/TorchTypes.td"

include "libtriton-core/Dialect/TritonRT/IR/TritonRTBase.td"

class TritonRT_Op<string mnemonic, list<Trait> traits = []>
    : Op<TritonRT_Dialect, mnemonic, traits>;

def LaunchIndx : AnyTypeOf<[Index, I32, I64]>;

def TritonRT_KernelOperandType : AnyTypeOf<[
  AnySignlessInteger, AnyFloat, Index, AnyMemRef, AnyTensor,
  Torch_ValueTensorType
]>;

def TritonRT_TritonKernelLaunchOp : TritonRT_Op<"triton_kernel_launch", [
  AttrSizedOperandSegments, AllTypesMatch<[
    "gridSizeX", "gridSizeY", "gridSizeZ", "blockSizeX", "blockSizeY",
    "blockSizeZ"
  ]>
]> {
  let summary = "Launch a Triton kernel entry point";
  let description = [{
    Models launching a Triton kernel symbol with explicit grid/block sizes.
    In addition to regular scalar and memref launch arguments, this op also
    supports builtin tensor and torch vtensor operands for staging initial
    TritonRT integration.
  }];

  let arguments = (ins SymbolRefAttr:$kernel,
                   LaunchIndx:$gridSizeX,
                   LaunchIndx:$gridSizeY,
                   LaunchIndx:$gridSizeZ,
                   LaunchIndx:$blockSizeX,
                   LaunchIndx:$blockSizeY,
                   LaunchIndx:$blockSizeZ,
                   Optional<I32>:$dynamicSharedMemorySize,
                   Variadic<TritonRT_KernelOperandType>:$kernelOperands);
  let results = (outs);

  let assemblyFormat = [{
    $kernel
    `blocks` `in` `(` $gridSizeX `,` $gridSizeY `,` $gridSizeZ `)`
    `threads` `in` `(` $blockSizeX `,` $blockSizeY `,` $blockSizeZ `)`
    (`dynamic_shared_memory_size` $dynamicSharedMemorySize^)?
    (`args` `(` $kernelOperands^ `:` type($kernelOperands) `)`)?
    attr-dict
    `:` type($gridSizeX)
  }];
}

#endif // LIBTRITON_CORE_DIALECT_TRITONRT_IR_TRITONRTOPS_TD_
