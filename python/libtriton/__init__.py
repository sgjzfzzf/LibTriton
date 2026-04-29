from __future__ import annotations

import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)

from .compile import (  # noqa: E402
    compile,
)

__all__ = [
    "compile",
]
