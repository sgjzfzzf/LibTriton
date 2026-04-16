import unittest

import torch

import libtriton


class TestTorchCompileExecutionEngineBackendCPU(unittest.TestCase):
    def test_add(self) -> None:
        @torch.compile(
            backend=libtriton.experimental_torch_mlir_execution_engine_backend,
            fullgraph=True,
        )
        def add_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        lhs = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        rhs = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
        out = add_fn(lhs, rhs)
        torch.testing.assert_close(out, lhs + rhs)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestTorchCompileExecutionEngineBackendCUDA(unittest.TestCase):
    def test_add(self) -> None:
        @torch.compile(
            backend=libtriton.experimental_torch_mlir_execution_engine_backend,
            fullgraph=True,
        )
        def add_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        lhs = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
        rhs = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="cuda")
        out = add_fn(lhs, rhs)
        torch.testing.assert_close(out.cpu(), (lhs + rhs).cpu())


if __name__ == "__main__":
    unittest.main()
