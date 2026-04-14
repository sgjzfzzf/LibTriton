import pathlib
from typing import Final

import numpy as np

import libtriton._C.libtriton_core._mlir_libs._libtritonCore as registration_ext
from libtriton._C.libtriton_core import capi_utils
from libtriton._C.libtriton_core import execution_engine, ir, passmanager

import tvm_ffi


MLIR_FILE: Final[pathlib.Path] = pathlib.Path(__file__).with_name("tvm_ffi_add.mlir")
TEST_FUNCTION: Final[str] = "tensor_add_kernel"


def _lower_to_llvm_dialect() -> ir.Module:
    ctx = ir.Context()
    with ctx:
        registration_ext.register_all_dialects(ctx)
        registration_ext.register_all_passes()
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
        shared_libs=[capi_utils.find_capi_runtime_library(registration_ext.__file__)],
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

    lhs = np.array([1, 2, 3, 4], dtype=np.int64)
    rhs = np.array([10, 20, 30, 40], dtype=np.int64)
    out = function(lhs, rhs)
    out_np = np.from_dlpack(out)
    np.testing.assert_array_equal(out_np, lhs + rhs)


if __name__ == "__main__":
    main()
