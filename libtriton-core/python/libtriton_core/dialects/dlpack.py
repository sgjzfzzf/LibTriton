"""DLPack dialect helpers."""

from .. import (
    dlpack_get_dl_context_type,
    dlpack_get_dl_data_type,
    dlpack_get_dl_managed_tensor_type,
    dlpack_get_dl_tensor_type,
    dlpack_is_dl_context_type,
    dlpack_is_dl_data_type,
    dlpack_is_dl_managed_tensor_type,
    dlpack_is_dl_tensor_type,
)

try:
    from ._dlpack_ops_gen import *  # noqa: F403
except ImportError:
    pass

__all__ = [
    "get_dl_context_type",
    "get_dl_data_type",
    "get_dl_tensor_type",
    "get_dl_managed_tensor_type",
    "is_dl_context_type",
    "is_dl_data_type",
    "is_dl_tensor_type",
    "is_dl_managed_tensor_type",
]


def get_dl_context_type(context):
    return dlpack_get_dl_context_type(context)


def get_dl_data_type(context):
    return dlpack_get_dl_data_type(context)


def get_dl_tensor_type(context):
    return dlpack_get_dl_tensor_type(context)


def get_dl_managed_tensor_type(context):
    return dlpack_get_dl_managed_tensor_type(context)


def is_dl_context_type(type):
    return dlpack_is_dl_context_type(type)


def is_dl_data_type(type):
    return dlpack_is_dl_data_type(type)


def is_dl_tensor_type(type):
    return dlpack_is_dl_tensor_type(type)


def is_dl_managed_tensor_type(type):
    return dlpack_is_dl_managed_tensor_type(type)
