from typing import Any, Dict, Final, List, Tuple

import torch
import tvm_ffi

from libtriton._C.libtriton_core import (
    capi_utils,
    compiler_utils,
    execution_engine,
    fx,
    ir,
    passmanager,
    register_all_dialects,
    register_all_passes,
)

from .transform import triton_graph_transform

_CPU_EXECUTION_ENGINE_PIPELINE: Final[str] = (
    "builtin.module("
    "one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map},"
    "emit-tvm-ffi-interface,"
    "convert-linalg-to-loops,"
    "convert-scf-to-cf,"
    "convert-to-llvm,"
    "convert-arith-to-llvm,"
    "convert-cf-to-llvm,"
    "finalize-memref-to-llvm{use-generic-functions=1},"
    "convert-func-to-llvm,"
    "reconcile-unrealized-casts"
    ")"
)

_CUDA_EXECUTION_ENGINE_PIPELINE: Final[str] = (
    "builtin.module("
    "one-shot-bufferize{{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map}},"
    "emit-tvm-ffi-interface,"
    "convert-linalg-to-parallel-loops,"
    "func.func(gpu-map-parallel-loops),"
    "convert-parallel-loops-to-gpu,"
    "gpu-kernel-outlining,"
    "lower-affine,"
    "convert-arith-to-llvm,"
    "convert-index-to-llvm,"
    "finalize-memref-to-llvm{{use-generic-functions=1}},"
    "nvvm-attach-target{{O=3 chip={chip}}},"
    "gpu.module(convert-gpu-to-nvvm{{index-bitwidth=64}}),"
    "gpu-to-llvm,"
    "gpu-module-to-binary{{format=fatbin}},"
    "convert-func-to-llvm,"
    "reconcile-unrealized-casts"
    ")"
)


def triton_graph_backend(
    gm: torch.fx.GraphModule, example_inputs: List[Any]
) -> torch.fx.GraphModule:
    _ = example_inputs
    modules: Dict[str, ir.Module] = {}
    gm = triton_graph_transform(gm)
    for name, child in gm.named_children():
        if name.startswith("submod_torch_"):
            modules[name] = fx.stateless_fx_import(child, model_name=name)
        elif name.startswith("submod_triton_"):
            ...
        else:
            raise ValueError(f"unknown submodule type: {name}")
    return gm


def experimental_torch_mlir_execution_engine_backend(
    gm: torch.fx.GraphModule, example_inputs: List[Any]
) -> Any:
    """Temporary experimental backend, planned to be replaced by a full backend."""
    model_name = "main"
    use_cuda = any(
        isinstance(arg, torch.Tensor) and arg.device.type == "cuda"
        for arg in example_inputs
    )
    module = fx.export_and_import(
        gm,
        *example_inputs,
        output_type=compiler_utils.OutputType.LINALG_ON_TENSORS,
        func_name=model_name,
    )

    with module.context:
        register_all_dialects(module.context)
        register_all_passes()
        module.operation.regions[0].blocks[0].operations[0].operation.attributes[
            "tvm_ffi.emit_tvm_ffi_interface"
        ] = ir.UnitAttr.get()
        if use_cuda:
            major, minor = torch.cuda.get_device_capability()
            cuda_pipeline = _CUDA_EXECUTION_ENGINE_PIPELINE.format(
                chip=f"sm_{major}{minor}"
            )
            passmanager.PassManager.parse(cuda_pipeline).run(module.operation)
            engine = execution_engine.ExecutionEngine(
                module,
                shared_libs=[
                    capi_utils.find_capi_runtime_library("cuda"),
                    capi_utils.find_mlir_cuda_runtime_library(),
                ],
            )
        else:
            passmanager.PassManager.parse(_CPU_EXECUTION_ENGINE_PIPELINE).run(
                module.operation
            )
            engine = execution_engine.ExecutionEngine(
                module,
                shared_libs=[capi_utils.find_capi_runtime_library("cpu")],
            )
        engine.initialize()
        ptr = engine.raw_lookup(f"__tvm_ffi_{model_name}")
        if ptr is None:
            raise RuntimeError(f"symbol not found: __tvm_ffi_{model_name}")
        fn = tvm_ffi.Function.__from_mlir_packed_safe_call__(
            ptr, keep_alive_object=engine
        )

    def compiled(*args: Any) -> Tuple[Any, ...]:
        result = fn(*args)
        if isinstance(result, tuple):
            values = result
        elif isinstance(result, list):
            values = tuple(result)
        else:
            values = (result,)

        converted = tuple(
            torch.from_dlpack(value)
            if hasattr(value, "__dlpack__") and not isinstance(value, torch.Tensor)
            else value
            for value in values
        )
        return converted

    return compiled
