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

# %%
using Libdl: dlext
# Load the shared library"
libpath = joinpath(@__DIR__, "../../target/debug/libsparse_ir_capi.$(dlext)")
#libpath = joinpath(@__DIR__, "../../libsparseir/backend/cxx/build/libsparseir.$(dlext)")

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

function spir_kernel_get_sve_hints_ngauss(kernel::Ptr{spir_kernel}, epsilon::Float64)
    status = Ref{Int32}(0)
    n_gauss = Ref{Int32}(0)
    ccall(
        (:spir_kernel_get_sve_hints_ngauss, libpath),
        Int32,
        (Ptr{spir_kernel}, Float64, Ptr{Int32}, Ref{Int32}),
        kernel, epsilon, n_gauss, status
    )
    return n_gauss[]
end

function spir_kernel_get_sve_hints_segments_x(kernel::Ptr{spir_kernel}, epsilon::Float64)
    status = Ref{Int32}(0)
    n_segments_x = Ref{Int32}(0)
    status = ccall(
        (:spir_kernel_get_sve_hints_segments_x, libpath),
        Float64,
        (Ptr{spir_kernel}, Float64, Ptr{Nothing}, Ptr{Int32}),
        kernel, epsilon, C_NULL, n_segments_x
    )

    segments_x = zeros(n_segments_x[] + 1)
    n_segments_x_out = Cint(n_segments_x[] + 1)
    ccall(
        (:spir_kernel_get_sve_hints_segments_x, libpath),
        Float64,
        (Ptr{spir_kernel}, Float64, Ptr{Float64}, Ptr{Int32}),
        kernel, epsilon, segments_x, Ref(n_segments_x_out)
    )
    return segments_x
end

function spir_kernel_get_sve_hints_segments_y(kernel::Ptr{spir_kernel}, epsilon::Float64)
    status = Ref{Int32}(0)
    n_segments_y = Ref{Int32}(0)
    # First call: get the number of segments
    status = ccall(
        (:spir_kernel_get_sve_hints_segments_y, libpath),
        Float64,
        (Ptr{spir_kernel}, Float64, Ptr{Nothing}, Ptr{Int32}),
        kernel, epsilon, C_NULL, n_segments_y
    )

    segments_y = zeros(n_segments_y[] + 1)
    n_segments_y_out = Cint(n_segments_y[] + 1)
    # Second call: get the actual segments
    ccall(
        (:spir_kernel_get_sve_hints_segments_y, libpath),
        Float64,
        (Ptr{spir_kernel}, Float64, Ptr{Float64}, Ptr{Int32}),
        kernel, epsilon, segments_y, Ref(n_segments_y_out)
    )
    return segments_y
end

# %%
let
    kernel = kernel_logistic_new(10.0)
    n_gauss = spir_kernel_get_sve_hints_ngauss(kernel, 1e-8)
    segments_x = spir_kernel_get_sve_hints_segments_x(kernel, 1e-8)
    segments_y = spir_kernel_get_sve_hints_segments_y(kernel, 1e-8)
    @assert length(segments_x) == 16
    @assert length(segments_y) == 21
end

# %%
let
    ngauss(epsilon) = epsilon >= 1e-8 ? 10 : 16
    lambda = 10.0
    epsilon = 1e-8
    full_hints = SparseIR.sve_hints(SparseIR.LogisticKernel(lambda), epsilon)
    n_gauss = ngauss(epsilon)
    @assert n_gauss == 10
    segs_x = SparseIR.segments_x(full_hints)
    segs_x = segs_x[segs_x.>=0]
    length(segs_x) == 16
    segs_y = SparseIR.segments_y(full_hints)
    segs_y = segs_y[segs_y.>=0]
    length(segs_y) == 21
end
