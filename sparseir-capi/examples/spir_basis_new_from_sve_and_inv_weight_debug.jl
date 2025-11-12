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
const libpath = joinpath(dirname(dirname(@__DIR__)), "target", "debug", "libsparseir_capi.$(dlext)")

# %%
lambda = 10.0
beta = 1.0
omega_max = lambda / beta
epsilon = 1e-8

# %%
# Opaque type
mutable struct spir_kernel end
mutable struct sve_result end
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

function sve_result_new(kernel, epsilon::Float64, cutoff::Float64, lmax, n_gauss, twork)
    status = Ref{Int32}(0)
    sve = ccall(
        (:spir_sve_result_new, libpath),
        Ptr{sve_result},
        (Ptr{spir_kernel}, Float64, Float64, Int32, Int32, Int32, Ref{Int32}),
        kernel, epsilon, cutoff, lmax, n_gauss, twork, status
    )

    if sve == C_NULL
        error("Failed to create SVE result: status = $(status[])")
    end

    return sve
end

function spir_funcs_from_piecewise_legendre(segments, n_segments, coeffs, nfuncs, order)
    status = Ref{Int32}(0)
    funcs = ccall(
        (:spir_funcs_from_piecewise_legendre, libpath),
        Ptr{Cvoid},
        (Ptr{Float64}, Int32, Ptr{Float64}, Int32, Int32, Ref{Int32}),
        segments, n_segments, coeffs, nfuncs, order, status
    )

    if funcs == C_NULL
        error("Failed to create funcs: status = $(status[])")
    end

    return funcs
end

function spir_basis_new_from_sve_and_inv_weight(statistics, beta, omega_max, epsilon, lambda, ypower, conv_radius, sve, inv_weight_funcs, max_size)
    status = Ref{Int32}(0)
    basis = ccall(
        (:spir_basis_new_from_sve_and_inv_weight, libpath),
        Ptr{Cvoid},
        (Int32, Float64, Float64, Float64, Float64, Int32, Float64, Ptr{sve_result}, Ptr{Cvoid}, Int32, Ref{Int32}),
        statistics, beta, omega_max, epsilon, lambda, ypower, conv_radius, sve, inv_weight_funcs, max_size, status
    )

    if basis == C_NULL
        error("Failed to create basis: status = $(status[])")
    end

    return basis
end

# %%
kernel = kernel_logistic_new(10.0)
sve = sve_result_new(kernel, epsilon, -1.0, -1, -1, -1)

# %%
n_segments = 1
segments = [-omega_max, omega_max]
coeffs = [1.0]
nfuncs = 1
order = 0
inv_weight_funcs = spir_funcs_from_piecewise_legendre(segments, n_segments, coeffs, nfuncs, order)

# %%
basis = spir_basis_new_from_sve_and_inv_weight(1, beta, omega_max, epsilon, lambda, 0, 1.0, sve, inv_weight_funcs, -1)

# %%
