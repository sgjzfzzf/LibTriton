import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)

from ._mlir_libs._libtritonCore import register_all_dialects, register_all_passes
from . import capi_utils, execution_engine, ir, passmanager, rewrite

try:
    from . import compiler_utils
except ImportError:
    compiler_utils = None

try:
    from . import fx
except ImportError:
    fx = None

__all__ = [
    "capi_utils",
    "compiler_utils",
    "execution_engine",
    "fx",
    "ir",
    "passmanager",
    "register_all_dialects",
    "register_all_passes",
    "rewrite",
]
