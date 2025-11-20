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
n_segments = 1
segments = [-1.0, 1.0]
coeffs = [1.0]
nfuncs = 1
order = 1

status = Ref{Cint}(-7)
funcs = ccall((:spir_funcs_from_piecewise_legendre, libpath),
    Ptr{Cvoid},
    (Ptr{Float64}, Cint, Ptr{Float64}, Cint, Cint, Ref{Cint}),
    pointer(segments), n_segments, pointer(coeffs), nfuncs, order, status)

println("status after spir_funcs_from_piecewise_legendre: ", status[])
println("funcs pointer: ", funcs)
println("funcs is null: ", funcs == C_NULL)

if status[] != 0
    error("Failed to compute functions: status = ", status[])
end

if funcs == C_NULL
    error("funcs is null even though status is ", status[])
end

size = Ref{Int32}(-1)
size_status = ccall(
    (:spir_funcs_get_size, libpath),
    Cint,
    (Ptr{Cvoid}, Ptr{Cint}),
    funcs, size
)

println("size_status: ", size_status)
@assert size_status == 0
@assert size[] == 1

# %%
