import pathlib
from typing import Final

import torch
import tvm_ffi

from libtriton.core import (
    capi_utils,
    ir,
    passmanager,
    execution_engine,
    register_all_dialects,
    register_all_passes,
)


MLIR_FILE: Final[pathlib.Path] = (
    pathlib.Path(__file__).parent / "Inputs" / "tvm_ffi_tensor_add.mlir"
)
TEST_FUNCTION: Final[str] = "tensor_add_kernel"


def _lower_to_llvm_dialect() -> ir.Module:
    ctx = ir.Context()
    with ctx:
        register_all_dialects(ctx)
        register_all_passes()
        module = ir.Module.parse(MLIR_FILE.read_text(encoding="utf-8"))
        pipeline = (
            "builtin.module("
            "emit-tvm-ffi-interface,"
            "convert-to-llvm,"
            "convert-index-to-llvm,"
            "convert-arith-to-llvm,"
            "finalize-memref-to-llvm,"
            "convert-func-to-llvm,"
            "reconcile-unrealized-casts"
            ")"
        )
        passmanager.PassManager.parse(pipeline).run(module.operation)
        return module


def _build_tvm_ffi_function(module: ir.Module) -> tvm_ffi.Function:
    engine = execution_engine.ExecutionEngine(
        module,
        shared_libs=[capi_utils.find_capi_runtime_library()],
    )
    function_ptr = engine.raw_lookup(f"__tvm_ffi_{TEST_FUNCTION}")
    if function_ptr is None:
        raise RuntimeError(f"function pointer not found: __tvm_ffi_{TEST_FUNCTION}")
    return tvm_ffi.Function.__from_mlir_packed_safe_call__(
        function_ptr,
        keep_alive_object=engine,
    )


def main() -> None:
    module = _lower_to_llvm_dialect()
    function = _build_tvm_ffi_function(module)

    lhs = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    rhs = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
    out = function(lhs, rhs)
    out_torch = torch.from_dlpack(out)
    torch.testing.assert_close(out_torch, lhs + rhs)


if __name__ == "__main__":
    main()
