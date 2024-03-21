"""
Provides the list of patches applied for files that are not exported inside google.

These are usually long-term duration patches that could not be applied in the previous copybara
workflow.
"""

excluded_files_patch_list = [
    "//third_party/triton/excluded:cl607293980.patch",
]
