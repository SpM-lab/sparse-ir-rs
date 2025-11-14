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
mutable struct spir_funcs end

function spir_funcs_get_size(funcs::Ptr{spir_funcs})
    sz = Ref{Int32}(0)
    status = ccall(
        (:spir_funcs_get_size, libpath),
        Int32,
        (Ptr{spir_funcs}, Ptr{Int32}),
        funcs, sz,
    )
    return sz[]
end

# %%
function spir_basis_get_uhat_full(basis::Ptr{spir_basis})
    status = Ref{Int32}(0)
    uhat_full = ccall(
        (:spir_basis_get_uhat_full, libpath),
        Ptr{spir_funcs},
        (Ptr{spir_basis}, Ref{Int32}),
        basis, status)
    if uhat_full == C_NULL
        error("Failed to create uhat: status = $(status[])")
    end
    return uhat_full
end

function spir_basis_get_uhat(basis::Ptr{spir_basis})
    status = Ref{Int32}(0)
    uhat = ccall(
        (:spir_basis_get_uhat, libpath),
        Ptr{spir_funcs},
        (Ptr{spir_basis}, Ref{Int32}),
        basis, status)
    if uhat == C_NULL
        error("Failed to create uhat: status = $(status[])")
    end
    return uhat
end

# %%
function spir_uhat_get_default_matsus(
    uhat::Ptr{spir_funcs},
    l::Integer,
    positive_only::Bool,
    mitigate::Bool
)
    points_buffer = Vector{Int64}()
    n_points_returned = Ref{Int32}(0)
    status = ccall(
        (:spir_uhat_get_default_matsus, libpath),
        Int32,
        (Ptr{spir_funcs}, Int32, Bool, Bool, Ptr{Int64}, Ptr{Int32}),
        uhat, Cint(l), positive_only, mitigate, pointer(points_buffer), n_points_returned
    )
    if status != 0
        error("Failed to get default Matsubara frequencies: status = $status")
    end
    # Return only the actual points that were written
    unsafe_wrap(Vector{Int64}, pointer(points_buffer), n_points_returned[])
end

# %%
let
    kernel = kernel_logistic_new(10.0)
    basis = spir_basis_new(1, 10.0, 1.0, 1e-6, kernel)
    uhat_full = spir_basis_get_uhat_full(basis)
    uhat = spir_basis_get_uhat(basis)
    uhat_size = spir_funcs_get_size(uhat)
    uhat_full_size = spir_funcs_get_size(uhat_full)
    @assert uhat_size == 10
    @assert uhat_full_size >= uhat_size

    let
        positive_only = true
        mitigate = false
        default_matsus_positive_only = spir_uhat_get_default_matsus(
            uhat, 10, positive_only, mitigate
        )
        println("default_matsus_positive_only: $default_matsus_positive_only")
        @assert default_matsus_positive_only == [3, 5, 7, 15]
    end

    let
        positive_only = false
        mitigate = false
        default_matsus = spir_uhat_get_default_matsus(
            uhat, 10, positive_only, mitigate
        )
        println("default_matsus: $default_matsus")
        @assert default_matsus == [
            -15, -7, -5, -3, 3, 5, 7, 15
        ]
    end
end

# %%
let
    kernel = SparseIR.LogisticKernel(10.0)
    basis = SparseIR.FiniteTempBasis{SparseIR.Fermionic}(10.0, 1.0, 1e-6)
    SparseIR.default_matsubara_sampling_points(
        basis.uhat, 10; fence=false, positive_only=true
    )
end

# %%
