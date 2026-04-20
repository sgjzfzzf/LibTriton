import unittest
from typing import Final

import torch
import triton
import triton.language as tl

from libtriton._C.libtriton_core import ir, register_all_dialects  # type: ignore[import-not-found]
from libtriton._C.libtriton_core.dialects import arith  # type: ignore[import-not-found]
from libtriton._C.libtriton_core.dialects import func  # type: ignore[import-not-found]
from libtriton._C.libtriton_core.dialects import gpu  # type: ignore[import-not-found]

_BLOCK_SIZE: Final[int] = 1024


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
        cubin = self.cubin

        self.assertIsNotNone(cubin)
        self.assertIsInstance(cubin, bytes)
        self.assertGreater(len(cubin), 0)
        self.assertEqual(cubin[:4], b"\x7fELF")

    def test_build_mlir_gpu_launch_module_from_cubin(self) -> None:
        cubin = self.cubin

        self.assertIsNotNone(cubin)
        self.assertIsInstance(cubin, bytes)
        cubin_hex = cubin.hex()

        ctx = ir.Context()
        with ctx, ir.Location.unknown():
            register_all_dialects(ctx)
            module = ir.Module.create()
            module.operation.attributes["gpu.container_module"] = ir.UnitAttr.get()
            index_type = ir.IndexType.get()
            i32_type = ir.IntegerType.get_signless(32)
            f32 = ir.F32Type.get()
            memref_4096xf32 = ir.MemRefType.get([4096], f32)

            gpu_object = ir.Attribute.parse(
                f'#gpu.object<#nvvm.target<chip = "{self.gpu_chip}">, "{cubin_hex}">'
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

                gpu.LaunchFuncOp(
                    kernel=[self.kernel_name, self.kernel_name],
                    grid_size=(grid_x, grid_y, grid_z),
                    block_size=(block_x, block_y, block_z),
                    kernel_operands=[
                        entry.arguments[0],
                        entry.arguments[1],
                        entry.arguments[2],
                    ],
                    dynamic_shared_memory_size=dynamic_shared_memory_size,
                )
                func.ReturnOp([])

        module_text = f"{module}"
        self.assertIn("gpu.launch_func", module_text)
        self.assertIn("gpu.binary", module_text)
        self.assertIn(f"@{self.kernel_name}::@{self.kernel_name}", module_text)


if __name__ == "__main__":
    unittest.main()
