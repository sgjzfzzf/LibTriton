import unittest
from typing import Final

import torch
import triton
import triton.language as tl

from libtriton._C.libtriton_core import (
    capi_utils,
    execution_engine,
    ir,
    passmanager,
    register_all_dialects,
    register_all_passes,
)  # type: ignore[import-not-found]
from libtriton._C.libtriton_core.dialects import arith, func, gpu  # type: ignore[import-not-found]

_BLOCK_SIZE: Final[int] = 1024
_GPU_BINARY_TO_LLVM_PIPELINE: Final[str] = (
    "builtin.module("
    "convert-arith-to-llvm,"
    "convert-index-to-llvm,"
    "finalize-memref-to-llvm{use-generic-functions=1},"
    "convert-func-to-llvm,"
    "gpu-to-llvm,"
    "reconcile-unrealized-casts"
    ")"
)


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestTritonAddKernelCubin(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda")
        cls.n_elements = 4096
        cls.x = torch.rand(cls.n_elements, device=cls.device)
        cls.y = torch.rand(cls.n_elements, device=cls.device)
        cls.output = torch.empty_like(cls.x)

        def grid(meta) -> tuple[int]:
            return (triton.cdiv(cls.n_elements, meta["BLOCK_SIZE"]),)

        cls.launch_grid = grid({"BLOCK_SIZE": _BLOCK_SIZE})

        cls.compiled_kernel = add_kernel[grid](
            cls.x, cls.y, cls.output, cls.n_elements, BLOCK_SIZE=_BLOCK_SIZE
        )
        cls.cubin = cls.compiled_kernel.asm.get("cubin")
        cls.kernel_name = cls.compiled_kernel.metadata.name
        cls.gpu_chip = f"sm_{cls.compiled_kernel.metadata.target.arch}"
        cls.block_size = (
            cls.compiled_kernel.metadata.num_warps
            * cls.compiled_kernel.metadata.warp_size,
            1,
            1,
        )
        cls.dynamic_shared_memory_size = cls.compiled_kernel.metadata.shared

    def test_capture_cubin_from_add_kernel(self) -> None:
        self.assertIsNotNone(self.cubin)
        self.assertIsInstance(self.cubin, bytes)
        self.assertGreater(len(self.cubin), 0)
        self.assertEqual(self.cubin[:4], b"\x7fELF")

    def _build_gpu_launch_module(self, ctx: ir.Context) -> ir.Module:
        with ctx, ir.Location.unknown():
            module = ir.Module.create()
            module.operation.attributes["gpu.container_module"] = ir.UnitAttr.get()
            index_type = ir.IndexType.get()
            i32_type = ir.IntegerType.get_signless(32)
            f32 = ir.F32Type.get()
            memref_4096xf32 = ir.MemRefType.get([4096], f32)

            cubin_mlir = "".join(f"\\{b:02X}" for b in self.cubin)
            gpu_object = ir.Attribute.parse(
                f'#gpu.object<#nvvm.target<chip = "{self.gpu_chip}">, "{cubin_mlir}">'
            )
            module.body.append(
                ir.Operation.create(
                    "gpu.binary",
                    attributes={
                        "sym_name": ir.StringAttr.get(self.kernel_name),
                        "objects": ir.ArrayAttr.get([gpu_object]),
                        "offloadingHandler": ir.Attribute.parse("#gpu.select_object"),
                    },
                )
            )

            fn_type = ir.FunctionType.get(
                [memref_4096xf32, memref_4096xf32, memref_4096xf32], []
            )
            fn = func.FuncOp("launch_add_kernel", fn_type)
            module.body.append(fn.operation)

            entry = ir.Block.create_at_start(
                fn.regions[0], [memref_4096xf32, memref_4096xf32, memref_4096xf32]
            )
            with ir.InsertionPoint(entry):
                (grid_x,) = self.launch_grid
                grid_x = arith.constant(index_type, grid_x)
                grid_y = arith.constant(index_type, 1)
                grid_z = arith.constant(index_type, 1)
                block_x, block_y, block_z = self.block_size
                block_x = arith.constant(index_type, block_x)
                block_y = arith.constant(index_type, block_y)
                block_z = arith.constant(index_type, block_z)
                dynamic_shared_memory_size = arith.constant(
                    i32_type, self.dynamic_shared_memory_size
                )
                arg0, arg1, arg2 = entry.arguments
                gpu.LaunchFuncOp(
                    kernel=[self.kernel_name, self.kernel_name],
                    grid_size=(grid_x, grid_y, grid_z),
                    block_size=(block_x, block_y, block_z),
                    kernel_operands=[arg0, arg1, arg2],
                    dynamic_shared_memory_size=dynamic_shared_memory_size,
                )
                func.ReturnOp([])

        return module

    def test_build_mlir_gpu_launch_module_from_cubin(self) -> None:
        ctx = ir.Context()
        register_all_dialects(ctx)
        module = self._build_gpu_launch_module(ctx)
        module_text = f"{module}"
        self.assertIn("gpu.launch_func", module_text)
        self.assertIn("gpu.binary", module_text)
        self.assertIn(f"@{self.kernel_name}::@{self.kernel_name}", module_text)

    def test_lower_gpu_launch_module_to_llvm_dialect(self) -> None:
        ctx = ir.Context()
        register_all_dialects(ctx)
        module = self._build_gpu_launch_module(ctx)
        with ctx:
            register_all_passes()
            passmanager.PassManager.parse(_GPU_BINARY_TO_LLVM_PIPELINE).run(
                module.operation
            )

        shared_libs = [
            capi_utils.find_capi_runtime_library("cuda"),
            capi_utils.find_mlir_cuda_runtime_library(),
        ]
        engine = execution_engine.ExecutionEngine(module, shared_libs=shared_libs)
        engine.initialize()
        fn_ptr = engine.raw_lookup("launch_add_kernel")
        self.assertIsNotNone(fn_ptr)


if __name__ == "__main__":
    unittest.main()
