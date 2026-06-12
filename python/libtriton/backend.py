from __future__ import annotations
from typing import Any, Callable, Final

import torch
import tvm_ffi

from libtriton._C.libtriton_core import (
    capi_utils,
    ir,
    passmanager,
    register_all_dialects,
    register_all_passes,
)
from libtriton._C.libtriton_core.dialects import func, tvm_ffi as tvm_ffi_d
from libtriton._C.libtriton_core.execution_engine import ExecutionEngine
from .importer import LibTritonFxImporter


class LibTritonGraphModule(object):
    """Compiles a torch function via Torch-MLIR and wraps with tvm_ffi."""

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fn: Final[Callable[..., Any]] = fn
        self.executor: Callable[..., Any] | None = None
        self.ctx: Final[ir.Context] = ir.Context()
        register_all_dialects(self.ctx)
        register_all_passes()

    def __call__(self, *args: Any) -> Any:
        self.executor = self.compile(*args)
        return self.executor(*args)

    def compile(self, *args: Any) -> Callable[..., Any]:
        # Step 1: Export the function to an FX GraphModule
        gm, _ = torch._dynamo.export(
            self.fn, aten_graph=True, assume_static_by_default=True
        )(*args)

        # Step 2: Import FX graph into MLIR using Torch-MLIR's FxImporter
        importer = LibTritonFxImporter(context=self.ctx)
        main_func = importer.import_stateless_graph(gm.graph, func_name="main")
        module = importer.module

        # Step 3: Add a tvm_ffi.func wrapper that calls func.func @main
        func_type = main_func.type

        with ir.InsertionPoint(module.body), ir.Location.unknown(self.ctx):
            ffi_func = tvm_ffi_d.func(self.fn.__name__, ir.TypeAttr.get(func_type))
            entry_block = ir.Block.create_at_start(ffi_func.body, func_type.inputs)
            with ir.InsertionPoint(entry_block):
                call_op = func.CallOp(
                    func_type.results,
                    "main",
                    entry_block.arguments,
                )
                tvm_ffi_d.return_(call_op)

        # Step 4: Lower Torch → GPU/LLVM via the Torch-to-LLVM pipeline
        with ir.Location.unknown(self.ctx):
            pm = passmanager.PassManager.parse("builtin.module(torch-to-llvm-pipeline)")
            pm.run(module.operation)

        # Step 5: JIT compile and wrap as a tvm_ffi function

        engine = ExecutionEngine(
            module,
            shared_libs=capi_utils.find_runtime_libraries(),
        )
        engine.initialize()

        symbol = f"__tvm_ffi_{self.fn.__name__}"
        ptr = engine.raw_lookup(symbol)
        assert ptr is not None, (
            f"symbol not found: {symbol}; "
            f"available: {[op.sym_name.value for op in module.body.operations if hasattr(op, 'sym_name')]}"
        )

        fn = tvm_ffi.Function.__from_mlir_packed_safe_call__(
            ptr, keep_alive_object=engine
        )

        def executor(*exec_args: Any) -> Any:

            def canonicalize(v: Any) -> Any:
                if isinstance(v, (list, tuple, tvm_ffi.Array)):
                    return type(v)(canonicalize(x) for x in v)
                elif hasattr(v, "__dlpack__") and not isinstance(v, torch.Tensor):
                    return torch.from_dlpack(v)
                else:
                    return v

            return canonicalize(fn(*exec_args))

        return executor
