from __future__ import annotations

import importlib
import typing

_core = importlib.import_module("libtriton._C.libtriton_core")


def __getattr__(name: str) -> typing.Any:
    return getattr(_core, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_core)))


__all__ = getattr(_core, "__all__", [])
