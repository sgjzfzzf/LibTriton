import pathlib
from typing import Final

_RUNTIME_BASENAME: Final[str] = "libLibTritonCoreRuntime.so"


def find_capi_runtime_library() -> str:
    module_dir = pathlib.Path(__file__).resolve().parent
    runtime_paths = sorted(
        (path for path in module_dir.rglob(_RUNTIME_BASENAME) if path.is_file()),
        key=lambda path: (len(path.parts), str(path)),
    )
    if not runtime_paths:
        raise RuntimeError(
            "missing LibTriton Runtime library under "
            f"{module_dir} via glob '**/{_RUNTIME_BASENAME}'"
        )
    return str(runtime_paths[0])
