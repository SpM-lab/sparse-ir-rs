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


# %%
def parse_functions_from_header(header_path):
    index = Index.create()
    translation_unit = index.parse(header_path, args=["-x", "c", "-DSPARSEIR_USE_EXTERN_FBLAS_PTR=ON", "-DSPARSEIR_USE_BLAS_ILP64=ON"])

    functions = dict()
    for child in translation_unit.cursor.get_children():
        if child.kind == CursorKind.FUNCTION_DECL:
            if child.spelling.startswith("_"):
                continue
            functions[child.spelling] = dict()
            functions[child.spelling]["result_type"] = child.result_type.spelling
            functions[child.spelling]["arguments"] = []
            for arg in child.get_arguments():
                functions[child.spelling]["arguments"].append(arg.type.spelling)

    return functions

rust_capi_header = "include/sparseir.h"
rust_capi_functions = parse_functions_from_header(rust_capi_header)
cxx_capi_header = "../../libsparseir/backend/cxx/include/sparseir/sparseir.h"
cxx_capi_functions = parse_functions_from_header(cxx_capi_header)

# %%
rust_capi_spellings = set(rust_capi_functions.keys())
cxx_capi_spellings = set(cxx_capi_functions.keys())

# %%
rust_capi_spellings.difference(cxx_capi_spellings)

# %%
cxx_capi_spellings.difference(rust_capi_spellings)

# %%
