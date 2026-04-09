"""TVM FFI dialect helpers."""

from .. import (
    tvm_ffi_get_any_type,
    tvm_ffi_get_object_handle_type,
    tvm_ffi_is_any_type,
    tvm_ffi_is_object_handle_type,
)

try:
    from ._tvm_ffi_ops_gen import *  # noqa: F403
except ImportError:
    pass

__all__ = [
    "get_any_type",
    "get_object_handle_type",
    "is_any_type",
    "is_object_handle_type",
]


def get_any_type(context):
    return tvm_ffi_get_any_type(context)


def get_object_handle_type(context):
    return tvm_ffi_get_object_handle_type(context)


def is_any_type(type):
    return tvm_ffi_is_any_type(type)


def is_object_handle_type(type):
    return tvm_ffi_is_object_handle_type(type)
