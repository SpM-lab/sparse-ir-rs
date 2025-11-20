# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Julia 1.12.1
#     language: julia
#     name: julia-1.12
# ---

# %%
using Libdl: dlext
# Load the shared library"
const libpath = joinpath(@__DIR__, "../../target/debug/libsparse_ir_capi.$(dlext)")

# %%
mutable struct spir_kernel end

"""
Create a Logistic kernel
"""
function kernel_logistic_new(lambda::Float64)
    status = Ref{Int32}(0)
    kernel = ccall(
        (:spir_logistic_kernel_new, libpath),
        Ptr{spir_kernel},
        (Float64, Ref{Int32}),
        lambda, status
    )

    if kernel == C_NULL
        error("Failed to create kernel: status = $(status[])")
    end

    return kernel
end

kernel = kernel_logistic_new(10.0)

xmin = Ref{Cdouble}(0.0)
xmax = Ref{Cdouble}(0.0)
ymin = Ref{Cdouble}(0.0)
ymax = Ref{Cdouble}(0.0)

ccall(
    (:spir_kernel_get_domain, libpath), Cint, (Ptr{spir_kernel}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
    kernel, xmin, xmax, ymin, ymax
)

@assert xmin[] == -1.0
@assert xmax[] == 1.0
@assert ymin[] == -1.0
@assert ymax[] == 1.0

# %%
