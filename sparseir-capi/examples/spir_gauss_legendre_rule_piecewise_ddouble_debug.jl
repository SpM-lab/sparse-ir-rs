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
n = 5
segments = [-1.0, 1.0]
n_segments = 1
x = zeros(n)
w = zeros(n)
status = Ref{Cint}(-7)

result = ccall((:spir_gauss_legendre_rule_piecewise_double, libpath),
    Cint,
    (Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}),
    n, segments, n_segments, x, w, status)

result

x[begin] >= -1.0
x[end] <= 1.0

for i in 1:n-1
    @assert x[begin+i] > x[begin+i-1]
end

weight_sum = 0.0
for i in 0:n-1
    @assert w[begin+i] > 0.0
    weight_sum += w[begin+i]
end

@assert abs(weight_sum - 2.0) < 1e-10

# %%
n = 5
segments = [-1.0, 1.0]
n_segments = 1
x_high = zeros(n)
x_low = zeros(n)
w_high = zeros(n)
w_low = zeros(n)
status = Ref{Cint}(-7)

result = ccall((:spir_gauss_legendre_rule_piecewise_ddouble, libpath),
    Cint,
    (Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}),
    n, segments, n_segments, x_high, x_low, w_high, w_low, status)

result

x_high[begin] >= -1.0
x_high[end] <= 1.0

for i in 1:n-1
    x_val = x_high[begin+i] + x_low[begin+i]
    x_prev = x_high[begin+i-1] + x_low[begin+i-1]
    @assert x_val > x_prev
end

weight_sum = 0.0
for i in 0:n-1
    w_val = w_high[begin+i] + w_low[begin+i]
    @assert w_val > 0.0
    weight_sum += w_val
end

@assert abs(weight_sum - 2.0) < 1e-10
