"""Provides the list of patches applied while importing Triton."""

patch_list = [
    "//third_party/triton:cl607293980.patch",  # long standing :(
    "//third_party/triton:sparse_dot_nvgpu.patch",
    "//third_party/triton/temporary:sparse_dot_base.patch",
    "//third_party/triton:sparse_dot_passes.patch",
]
