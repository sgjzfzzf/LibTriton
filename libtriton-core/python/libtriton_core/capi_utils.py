import pathlib
from typing import Final, Literal

_RUNTIME_BASENAMES: Final[dict[str, str]] = {
    "cpu": "libLibTritonCoreRuntimeCPU.so",
    "cuda": "libLibTritonCoreRuntimeCUDA.so",
}


def find_capi_runtime_library(runtime_type: Literal["cpu", "cuda"] = "cpu") -> str:
    """Find the LibTriton Runtime library.

    Args:
        runtime_type: "cpu" for CPU runtime, "cuda" for CUDA runtime

    Returns:
        Path to the runtime library

    Raises:
        RuntimeError: If the runtime library is not found
    """
    basename = _RUNTIME_BASENAMES[runtime_type]

    module_dir = pathlib.Path(__file__).resolve().parent
    runtime_lib = module_dir / "_runtime_libs" / basename
    if not runtime_lib.is_file():
        raise RuntimeError(
            f"missing LibTriton {runtime_type.upper()} Runtime library: {runtime_lib}"
        )
    return str(runtime_lib)


def find_mlir_cuda_runtime_library() -> str:
    """Find the MLIR CUDA runtime library.

    Returns:
        Path to libmlir_cuda_runtime.so

    Raises:
        RuntimeError: If the library is not found
    """
    module_dir = pathlib.Path(__file__).resolve().parent
    cuda_runtime_lib = module_dir / "_runtime_libs" / "libmlir_cuda_runtime.so"
    if not cuda_runtime_lib.is_file():
        raise RuntimeError(f"missing MLIR CUDA runtime library: {cuda_runtime_lib}")
    return str(cuda_runtime_lib)
