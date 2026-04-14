import pathlib

_RUNTIME_BASENAME = "libLibTritonCoreRuntime.so"


def find_capi_runtime_library(anchor_file: str | pathlib.Path) -> str:
    runtime_path = pathlib.Path(anchor_file).resolve().with_name(_RUNTIME_BASENAME)
    if not runtime_path.is_file():
        raise RuntimeError(f"missing LibTriton Runtime library at {runtime_path}")
    return str(runtime_path)
