import libtriton_core


assert hasattr(libtriton_core, "register_all_dialects")
assert hasattr(libtriton_core, "register_all_passes")
assert hasattr(libtriton_core, "dlpack_get_dl_context_type")
assert hasattr(libtriton_core, "tvm_ffi_get_any_type")
