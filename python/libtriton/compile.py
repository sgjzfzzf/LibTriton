from typing import Any, Callable

from .backend import TritonGraphModule


def compile(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Callable[..., Any]:
    """Compile a Python function with example inputs and return a cached executor."""
    gm: TritonGraphModule = TritonGraphModule(fn)
    gm.compile(*args, **kwargs)
    return gm.executor
