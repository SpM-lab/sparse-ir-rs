# -*- coding: utf-8 -*-
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
Pkg.add(name="SparseIR", version="1.1.4")
import SparseIR
using LinearAlgebra

# %%
using Libdl: dlext
# Load the shared library"
libpath = joinpath(@__DIR__, "../../target/debug/libsparseir_capi.$(dlext)")

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

function spir_gauss_legendre_rule_piecewise_double!(
    n_gauss::Integer,
    segments::Vector{Float64},
    x::Vector{Float64},
    w::Vector{Float64},
)
    status = Ref{Int32}(0)
    n_segments = length(segments) - 1

    return ccall(
        (:spir_gauss_legendre_rule_piecewise_double, libpath),
        Int32,
        (Int32, Ptr{Float64}, Int32, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}),
        Cint(n_gauss), segments, Cint(n_segments), x, w, status
    )
end

mutable struct spir_sve_result end
function spir_sve_result_from_matrix_centrosymmetric(
    K_even_high::Matrix{Float64},
    K_odd_high::Matrix{Float64},
    nx::Integer,
    ny::Integer,
    segments_x::Vector{Float64},
    segments_y::Vector{Float64},
    n_gauss::Integer,
    epsilon::Float64,
)
    status = Ref{Int32}(0)
    order = 1
    # n_segments_x is the number of segments (boundary points - 1)
    # segments_x has n_segments_x + 1 elements
    n_segments_x = length(segments_x) - 1
    n_segments_y = length(segments_y) - 1
    sve_result = ccall(
        (:spir_sve_result_from_matrix_centrosymmetric, libpath),
        Ptr{spir_sve_result},
        (
            Ptr{Float64}, Ptr{Nothing},
            Ptr{Float64}, Ptr{Nothing},
            Int32, Int32, Int32,
            Ptr{Float64}, Int32,
            Ptr{Float64}, Int32,
            Int32, Float64,
            Ptr{Int32}
        ),
        vec(K_even_high), C_NULL,
        vec(K_odd_high), C_NULL,
        nx, ny, 1,
        segments_x, n_segments_x,
        segments_y, n_segments_y,
        n_gauss, epsilon, status
    )
    if status[] != 0
        error("Failed to create SVE result: status = $(status[])")
    end
    return sve_result
end

function spir_sve_result_get_size(sve_result::Ptr{spir_sve_result})
    status = Ref{Int32}(0)
    size = Ref{Int32}(0)
    ccall(
        (:spir_sve_result_get_size, libpath),
        Int32,
        (Ptr{spir_sve_result}, Ptr{Int32}, Ptr{Int32}),
        sve_result, size, status
    )
    if status[] != 0
        error("Failed to get SVE result size: status = $(status[])")
    end
    return size[]
end

function spir_sve_result_get_svals(sve_result::Ptr{spir_sve_result})
    status = Ref{Int32}(0)
    svals_size = spir_sve_result_get_size(sve_result)
    svals = zeros(svals_size)
    ccall(
        (:spir_sve_result_get_svals, libpath),
        Int32,
        (Ptr{spir_sve_result}, Ptr{Float64}, Ptr{Int32}),
        sve_result, svals, status
    )
    if status[] != 0
        error("Failed to get SVE result svals: status = $(status[])")
    end
    return svals
end

svals_capi = let
    kernel = kernel_logistic_new(10.0)
    n_gauss = spir_kernel_get_sve_hints_ngauss(kernel, 1e-8)
    segments_x = spir_kernel_get_sve_hints_segments_x(kernel, 1e-8)
    segments_y = spir_kernel_get_sve_hints_segments_y(kernel, 1e-8)
    n_segments_x = length(segments_x) - 1
    n_segments_y = length(segments_y) - 1
    # Total number of Gauss points = n_gauss points per segment * number of segments
    nx = n_gauss * n_segments_x
    ny = n_gauss * n_segments_y
    x = zeros(nx)
    w_x = zeros(nx)
    y = zeros(ny)
    w_y = zeros(ny)
    spir_gauss_legendre_rule_piecewise_double!(
        n_gauss, segments_x, x, w_x
    )
    spir_gauss_legendre_rule_piecewise_double!(
        n_gauss, segments_y, y, w_y
    )

    K_even_high = zeros(nx, ny)
    K_odd_high = zeros(nx, ny)
    for i in 1:nx
        for j in 1:ny
            # Simple test: even part is symmetric, odd part is antisymmetric
            if i == j
                K_even_high[i, j] = sqrt(w_x[i] * w_y[j])
                K_odd_high[i, j] = sqrt(w_x[i] * w_y[j]) * 0.5 # Smaller odd part
            else
                K_even_high[i, j] = 0.0
                K_odd_high[i, j] = 0.0
            end
        end
    end

    sve_result = spir_sve_result_from_matrix_centrosymmetric(
        K_even_high, K_odd_high, nx, ny, segments_x, segments_y, n_gauss, 1e-8
    )
    svals = spir_sve_result_get_svals(sve_result)
end

# %%
function _postprocess(n_gauss, gauss_rule_x, gauss_rule_y, segs_x, segs_y, w_x, w_y, u, s, v)
    s = Float64.(s)
    u_x = u ./ sqrt.(w_x)
    v_y = v ./ sqrt.(w_y)

    u_x = reshape(u_x, (n_gauss, length(segs_x) - 1, length(s)))
    v_y = reshape(v_y, (n_gauss, length(segs_y) - 1, length(s)))

    cmat_x = SparseIR.legendre_collocation(gauss_rule_x)
    cmat_y = SparseIR.legendre_collocation(gauss_rule_y)
    # Reshape u_x from (n_gauss, n_segments, n_singular_values) to (nx, n_singular_values)
    u_x_reshaped = reshape(u_x, (size(u_x, 1) * size(u_x, 2), size(u_x, 3)))
    # Apply collocation matrix: (nx, nx) * (nx, n_singular_values) -> (nx, n_singular_values)
    u_data = cmat_x * u_x_reshaped
    # Reshape back to (n_gauss, n_segments, n_singular_values) for PiecewiseLegendrePolyVector
    u_data = reshape(u_data, (size(u_x, 1), size(u_x, 2), size(u_x, 3)))

    v_y_reshaped = reshape(v_y, (size(v_y, 1) * size(v_y, 2), size(v_y, 3)))
    v_data = cmat_y * v_y_reshaped
    v_data = reshape(v_data, (size(v_y, 1), size(v_y, 2), size(v_y, 3)))

    dsegs_x = diff(segs_x)
    dsegs_y = diff(segs_y)
    u_data .*= sqrt.(0.5 .* reshape(dsegs_x, (1, :)))
    v_data .*= sqrt.(0.5 .* reshape(dsegs_y, (1, :)))

    # Construct polynomials
    ulx = SparseIR.PiecewiseLegendrePolyVector(Float64.(u_data), Float64.(segs_x))
    vly = SparseIR.PiecewiseLegendrePolyVector(Float64.(v_data), Float64.(segs_y))
    SparseIR.canonicalize!(ulx, vly)
    return ulx, s, vly
end

function postprocess(
    full_hints,
    n_gauss, gauss_rule_x, gauss_rule_y, segs_x, segs_y, w_x, w_y,
    u_even, s_even, v_even,
    u_odd, s_odd, v_odd
)
    u_even, s_even, v_even = _postprocess(n_gauss, gauss_rule_x, gauss_rule_y, segs_x, segs_y, w_x, w_y, u_even, s_even, v_even)
    u_odd, s_odd, v_odd = _postprocess(n_gauss, gauss_rule_x, gauss_rule_y, segs_x, segs_y, w_x, w_y, u_odd, s_odd, v_odd)

    # Merge two sets
    u = SparseIR.PiecewiseLegendrePolyVector([u_even; u_odd])
    v = SparseIR.PiecewiseLegendrePolyVector([v_even; v_odd])
    s = [s_even; s_odd]
    signs = [fill(1, length(s_even)); fill(-1, length(s_odd))]

    # Sort: now for totally positive kernels like defined in this module,
    # this strictly speaking is not necessary as we know that the even/odd
    # functions intersperse.
    sort = sortperm(s; rev=true)
    u = u[sort]
    v = v[sort]
    s = s[sort]
    signs = signs[sort]

    # Extend to the negative side
    u_complete = similar(u)
    v_complete = similar(v)
    segs_x = SparseIR.segments_x(full_hints)
    segs_y = SparseIR.segments_y(full_hints)

    poly_flip_x = (-1) .^ range(0; length=size(first(u).data, 1))
    for i in eachindex(u, v)
        u_pos_data = u[i].data / sqrt(2)
        v_pos_data = v[i].data / sqrt(2)

        u_neg_data = reverse(u_pos_data; dims=2) .* poly_flip_x * signs[i]
        v_neg_data = reverse(v_pos_data; dims=2) .* poly_flip_x * signs[i]
        u_data = hcat(u_neg_data, u_pos_data)
        v_data = hcat(v_neg_data, v_pos_data)
        u_complete[i] = SparseIR.PiecewiseLegendrePoly(u_data, segs_x, i - 1; symm=signs[i])
        v_complete[i] = SparseIR.PiecewiseLegendrePoly(v_data, segs_y, i - 1; symm=signs[i])
    end

    u_complete, s, v_complete
end

let
    ngauss(epsilon) = epsilon >= 1e-8 ? 10 : 16

    lambda = 10.0
    epsilon = 1e-8
    full_hints = SparseIR.sve_hints(SparseIR.LogisticKernel(lambda), epsilon)
    n_gauss = ngauss(epsilon)
    @assert n_gauss == 10
    segments_x = SparseIR.segments_x(full_hints)
    segments_x = segments_x[segments_x.>=0]
    segments_y = SparseIR.segments_y(full_hints)
    segments_y = segments_y[segments_y.>=0]

    n_segments_x = length(segments_x) - 1
    n_segments_y = length(segments_y) - 1
    nx = n_gauss * n_segments_x
    ny = n_gauss * n_segments_y

    rule_x = SparseIR.legendre(n_gauss)
    rule_x = SparseIR.piecewise(rule_x, segments_x)

    rule_y = SparseIR.legendre(n_gauss)
    rule_y = SparseIR.piecewise(rule_y, segments_y)

    x = rule_x.x
    w_x = rule_x.w
    y = rule_y.x
    w_y = rule_y.w

    K_even_high = zeros(nx, ny)
    K_odd_high = zeros(nx, ny)
    for i in 1:nx
        for j in 1:ny
            # Simple test: even part is symmetric, odd part is antisymmetric
            if i == j
                K_even_high[i, j] = sqrt(w_x[i] * w_y[j])
                K_odd_high[i, j] = sqrt(w_x[i] * w_y[j]) * 0.5 # Smaller odd part
            else
                K_even_high[i, j] = 0.0
                K_odd_high[i, j] = 0.0
            end
        end
    end

    u_even, s_even, v_even = svd(K_even_high)
    u_odd, s_odd, v_odd = svd(K_odd_high)

    u, svals_julia, v = postprocess(
        full_hints,
        n_gauss, rule_x, rule_y, segments_x, segments_y, w_x, w_y,
        u_even, s_even, v_even,
        u_odd, s_odd, v_odd
    )
    @assert svals_julia ≈ svals_capi
end
