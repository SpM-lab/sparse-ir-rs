# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from clang.cindex import Index, CursorKind

index = Index.create()
filename = "include/sparseir.h"
translation_unit = index.parse(filename, args=["-x", "c"])

for child in translation_unit.cursor.get_children():
    functions = dict()
    if child.kind == CursorKind.FUNCTION_DECL:
        functions[child.spelling] = dict()
        functions[child.spelling]["result_type"] = child.result_type.spelling
        functions[child.spelling]["arguments"] = []
        for arg in child.get_arguments():
            functions[child.spelling]["arguments"].append(arg.type.spelling)
        print(child.spelling, functions[child.spelling])

# %%
