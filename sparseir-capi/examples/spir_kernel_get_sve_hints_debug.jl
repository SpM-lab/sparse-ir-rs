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
const libpath = joinpath(@__DIR__, "../../target/debug/libsparseir_capi.$(dlext)")

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

# %%
epsilon_coarse = 1e-6
epsilon_fine = 1e-10

ngauss_coarse = Ref{Cint}(0)
ngauss_fine = Ref{Cint}(0)

ccall(
    (:spir_kernel_get_sve_hints_ngauss, libpath),
    Cint,
    (Ptr{spir_kernel}, Cdouble, Ptr{Cint}),
    kernel, epsilon_coarse, ngauss_coarse
)
@assert ngauss_coarse[] == 10
ccall(
    (:spir_kernel_get_sve_hints_ngauss, libpath),
    Cint,
    (Ptr{spir_kernel}, Cdouble, Ptr{Cint}),
    kernel, epsilon_fine, ngauss_fine
)
@assert ngauss_fine[] == 16

# %%
epsilon = 1e-8

nsvals = Ref{Cint}(0)
ccall(
    (:spir_kernel_get_sve_hints_nsvals, libpath),
    Cint,
    (Ptr{spir_kernel}, Cdouble, Ptr{Cint}),
    kernel, epsilon, nsvals
)

@assert nsvals[] > 0
