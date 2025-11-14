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
using Pkg;
Pkg.activate(temp=true);
Pkg.add("SparseIR")
import SparseIR

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

# %%
mutable struct spir_basis end
mutable struct spir_sve_result end

function spir_basis_new(
    statistics::Integer,
    beta::Real,
    omega_max::Real,
    epsilon::Real,
    k::Ptr{spir_kernel},
)
    status = Ref{Int32}(-1)
    sve = C_NULL
    max_size = -1
    basis = ccall(
        (:spir_basis_new, libpath),
        Ptr{spir_basis},
        (Int32, Float64, Float64, Float64, Ptr{spir_kernel}, Ptr{spir_sve_result}, Int32, Ptr{Int32}),
        Cint(statistics), beta, omega_max, epsilon, k, sve, max_size, status
    )
    if basis == C_NULL
        error("Failed to create basis: status = $(status[])")
    end
    return basis
end

# %%
let
    kernel = kernel_logistic_new(10.0)
    basis = spir_basis_new(1, 10.0, 1.0, 1e-6, kernel)
end

# %%
let
    kernel = SparseIR.LogisticKernel(10.0)
    basis = SparseIR.FiniteTempBasis{SparseIR.Fermionic}(10.0, 1.0, 1e-6)
    basis.uhat_full
end
