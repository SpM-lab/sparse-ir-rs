#!/usr/bin/env julia
"""
Example: Using SparseIR C-API from Julia

This demonstrates how to call the SparseIR Rust library from Julia.
"""

using Libdl: dlext
# Load the shared library
const libpath = "../../target/debug/libsparse_ir_capi.$(dlext)"  # macOS

# Error codes (compatible with libsparseir)
const SPIR_COMPUTATION_SUCCESS = Int32(0)
const SPIR_GET_IMPL_FAILED = Int32(-1)
const SPIR_INVALID_DIMENSION = Int32(-2)
const SPIR_INPUT_DIMENSION_MISMATCH = Int32(-3)
const SPIR_OUTPUT_DIMENSION_MISMATCH = Int32(-4)
const SPIR_NOT_SUPPORTED = Int32(-5)
const SPIR_INVALID_ARGUMENT = Int32(-6)
const SPIR_INTERNAL_ERROR = Int32(-7)

# Aliases
const SPIR_SUCCESS = SPIR_COMPUTATION_SUCCESS

# Opaque type
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

"""
Create a RegularizedBose kernel
"""
function kernel_regularized_bose_new(lambda::Float64)
    status = Ref{Int32}(0)
    kernel = ccall(
        (:spir_reg_bose_kernel_new, libpath),
        Ptr{spir_kernel},
        (Float64, Ref{Int32}),
        lambda, status
    )

    if kernel == C_NULL
        error("Failed to create kernel: status = $(status[])")
    end

    return kernel
end

"""
Release a kernel
"""
function kernel_release(kernel::Ptr{spir_kernel})
    ccall(
        (:spir_kernel_release, libpath),
        Cvoid,
        (Ptr{spir_kernel},),
        kernel
    )
end

"""
Get lambda parameter
"""
function kernel_lambda(kernel::Ptr{spir_kernel})
    lambda = Ref{Float64}()
    status = ccall(
        (:spir_kernel_get_lambda, libpath),
        Int32,
        (Ptr{spir_kernel}, Ref{Float64}),
        kernel, lambda
    )

    if status != SPIR_SUCCESS
        error("Failed to get lambda: status = $status")
    end

    return lambda[]
end

"""
Compute kernel value K(x, y)
"""
function kernel_compute(kernel::Ptr{spir_kernel}, x::Float64, y::Float64)
    result = Ref{Float64}()
    status = ccall(
        (:spir_kernel_compute, libpath),
        Int32,
        (Ptr{spir_kernel}, Float64, Float64, Ref{Float64}),
        kernel, x, y, result
    )

    if status != SPIR_SUCCESS
        error("Failed to compute kernel: status = $status")
    end

    return result[]
end

# ============================================================================
# Main test
# ============================================================================

println("🧪 Testing SparseIR C-API from Julia")
println("=" ^ 50)

# Test 1: Logistic Kernel
println("\n📝 Test 1: Logistic Kernel")
kernel = kernel_logistic_new(10.0)
println("✅ Created kernel")

lambda = kernel_lambda(kernel)
println("   λ = $lambda")

value = kernel_compute(kernel, 0.5, 0.5)
println("   K(0.5, 0.5) = $value")

kernel_release(kernel)
println("✅ Released kernel")

# Test 2: RegularizedBose Kernel
println("\n📝 Test 2: RegularizedBose Kernel")
kernel = kernel_regularized_bose_new(10.0)
println("✅ Created kernel")

lambda = kernel_lambda(kernel)
println("   λ = $lambda")

value = kernel_compute(kernel, 0.5, 0.5)
println("   K(0.5, 0.5) = $value")

kernel_release(kernel)
println("✅ Released kernel")

# Test 3: Compute multiple values
println("\n📝 Test 3: Compute kernel on grid")
kernel = kernel_logistic_new(10.0)

x_grid = range(-1, 1, length=5)
y_grid = range(-1, 1, length=5)

println("   Grid: x ∈ [-1, 1], y ∈ [-1, 1]")
for x in x_grid
    for y in y_grid
        val = kernel_compute(kernel, x, y)
        print("$(round(val, digits=4)) ")
    end
    println()
end

kernel_release(kernel)

# Test 4: Error handling
println("\n📝 Test 4: Error handling")
try
    kernel = kernel_logistic_new(-1.0)  # Invalid lambda
    println("❌ Should have failed")
catch e
    println("✅ Correctly caught error: $e")
end

# Test 5: SVE computation
println("\n📝 Test 5: SVE Computation")
kernel = kernel_logistic_new(10.0)

# Compute SVE
status = Ref{Int32}(0)
sve = ccall(
    (:spir_sve_result_new, libpath),
    Ptr{Cvoid},
    (Ptr{Cvoid}, Float64, Float64, Int32, Int32, Int32, Ref{Int32}),
    kernel, 1e-6, -1.0, -1, -1, -1, status
)

if sve == C_NULL || status[] != SPIR_COMPUTATION_SUCCESS
    error("Failed to compute SVE: status = $(status[])")
end

# Get SVE size
sve_size = Ref{Int32}(0)
status_size = ccall(
    (:spir_sve_result_get_size, libpath),
    Int32,
    (Ptr{Cvoid}, Ref{Int32}),
    sve, sve_size
)

if status_size != SPIR_COMPUTATION_SUCCESS
    error("Failed to get SVE size: status = $status_size")
end

println("✅ Computed SVE")
println("   Size: $(sve_size[])")

# Get singular values
svals = Vector{Float64}(undef, sve_size[])
status_svals = ccall(
    (:spir_sve_result_get_svals, libpath),
    Int32,
    (Ptr{Cvoid}, Ptr{Float64}),
    sve, svals
)

if status_svals != SPIR_COMPUTATION_SUCCESS
    error("Failed to get singular values: status = $status_svals")
end

println("   First 5 singular values:")
for i in 1:min(5, length(svals))
    println("     s[$i] = $(svals[i])")
end

# Test truncation
status_truncate = Ref{Int32}(0)
sve_truncated = ccall(
    (:spir_sve_result_truncate, libpath),
    Ptr{Cvoid},
    (Ptr{Cvoid}, Float64, Int32, Ref{Int32}),
    sve, 1e-4, div(sve_size[], 2), status_truncate
)

if sve_truncated == C_NULL || status_truncate[] != SPIR_COMPUTATION_SUCCESS
    error("Failed to truncate SVE: status = $(status_truncate[])")
end

size_truncated = Ref{Int32}(0)
ccall(
    (:spir_sve_result_get_size, libpath),
    Int32,
    (Ptr{Cvoid}, Ref{Int32}),
    sve_truncated, size_truncated
)

println("   Truncated size: $(size_truncated[])")

# Cleanup
ccall((:spir_sve_result_release, libpath), Cvoid, (Ptr{Cvoid},), sve_truncated)
ccall((:spir_sve_result_release, libpath), Cvoid, (Ptr{Cvoid},), sve)
kernel_release(kernel)

# Test 6: Basis construction
println("\n📝 Test 6: Basis Construction")
kernel = kernel_logistic_new(10.0)

# Create basis
status_basis = Ref{Int32}(0)
basis = ccall(
    (:spir_basis_new, libpath),
    Ptr{Cvoid},
    (Int32, Float64, Float64, Float64, Ptr{Cvoid}, Ptr{Cvoid}, Int32, Ref{Int32}),
    1,      # Fermionic
    10.0,   # beta
    1.0,    # omega_max
    1e-6,   # epsilon
    kernel, # kernel
    C_NULL, # sve (will compute)
    -1,     # max_size
    status_basis
)

if basis == C_NULL || status_basis[] != SPIR_COMPUTATION_SUCCESS
    error("Failed to create basis: status = $(status_basis[])")
end

# Get basis size
basis_size = Ref{Int32}(0)
status_size = ccall(
    (:spir_basis_get_size, libpath),
    Int32,
    (Ptr{Cvoid}, Ref{Int32}),
    basis, basis_size
)

if status_size != SPIR_COMPUTATION_SUCCESS
    error("Failed to get basis size: status = $status_size")
end

println("✅ Created basis")
println("   Size: $(basis_size[])")

# Get statistics
stats = Ref{Int32}(0)
status_stats = ccall(
    (:spir_basis_get_stats, libpath),
    Int32,
    (Ptr{Cvoid}, Ref{Int32}),
    basis, stats
)
println("   Statistics: $(stats[] == 1 ? "Fermionic" : "Bosonic")")

# Get tau sampling points
n_taus = Ref{Int32}(0)
ccall(
    (:spir_basis_get_n_default_taus, libpath),
    Int32,
    (Ptr{Cvoid}, Ref{Int32}),
    basis, n_taus
)

tau_points = Vector{Float64}(undef, n_taus[])
ccall(
    (:spir_basis_get_default_taus, libpath),
    Int32,
    (Ptr{Cvoid}, Ptr{Float64}),
    basis, tau_points
)

println("   Tau sampling points: $(n_taus[])")
println("   First 3: $(tau_points[1:min(3, end)])")

# Get Matsubara sampling points
n_matsus = Ref{Int32}(0)
ccall(
    (:spir_basis_get_n_default_matsus, libpath),
    Int32,
    (Ptr{Cvoid}, Bool, Ref{Int32}),
    basis, true, n_matsus  # positive_only = true
)

matsu_points = Vector{Int64}(undef, n_matsus[])
ccall(
    (:spir_basis_get_default_matsus, libpath),
    Int32,
    (Ptr{Cvoid}, Bool, Ptr{Int64}),
    basis, true, matsu_points
)

println("   Matsubara sampling points (positive): $(n_matsus[])")
println("   First 3: $(matsu_points[1:min(3, end)])")

# Cleanup
ccall((:spir_basis_release, libpath), Cvoid, (Ptr{Cvoid},), basis)
kernel_release(kernel)

# Test 7: Funcs API
println("\n📝 Test 7: Funcs API (u, v, uhat)")
kernel = kernel_logistic_new(10.0)

# Create basis
status_basis = Ref{Int32}(0)
basis = ccall(
    (:spir_basis_new, libpath),
    Ptr{Cvoid},
    (Int32, Float64, Float64, Float64, Ptr{Cvoid}, Ptr{Cvoid}, Int32, Ref{Int32}),
    1,      # Fermionic
    10.0,   # beta
    1.0,    # omega_max
    1e-6,   # epsilon
    kernel,
    C_NULL,
    -1,
    status_basis
)

if basis == C_NULL || status_basis[] != SPIR_COMPUTATION_SUCCESS
    error("Failed to create basis: status = $(status_basis[])")
end

# Get u funcs (imaginary-time basis functions)
status_u = Ref{Int32}(0)
u_funcs = ccall(
    (:spir_basis_get_u, libpath),
    Ptr{Cvoid},
    (Ptr{Cvoid}, Ref{Int32}),
    basis, status_u
)

if u_funcs == C_NULL || status_u[] != SPIR_COMPUTATION_SUCCESS
    error("Failed to get u funcs: status = $(status_u[])")
end

println("✅ Got u funcs (imaginary-time basis functions)")

# Get v funcs (real-frequency basis functions)
status_v = Ref{Int32}(0)
v_funcs = ccall(
    (:spir_basis_get_v, libpath),
    Ptr{Cvoid},
    (Ptr{Cvoid}, Ref{Int32}),
    basis, status_v
)

if v_funcs == C_NULL || status_v[] != SPIR_COMPUTATION_SUCCESS
    error("Failed to get v funcs: status = $(status_v[])")
end

println("✅ Got v funcs (real-frequency basis functions)")

# Get uhat funcs (Matsubara-frequency basis functions)
status_uhat = Ref{Int32}(0)
uhat_funcs = ccall(
    (:spir_basis_get_uhat, libpath),
    Ptr{Cvoid},
    (Ptr{Cvoid}, Ref{Int32}),
    basis, status_uhat
)

if uhat_funcs == C_NULL || status_uhat[] != SPIR_COMPUTATION_SUCCESS
    error("Failed to get uhat funcs: status = $(status_uhat[])")
end

println("✅ Got uhat funcs (Matsubara-frequency basis functions)")

# Cleanup
ccall((:spir_funcs_release, libpath), Cvoid, (Ptr{Cvoid},), u_funcs)
ccall((:spir_funcs_release, libpath), Cvoid, (Ptr{Cvoid},), v_funcs)
ccall((:spir_funcs_release, libpath), Cvoid, (Ptr{Cvoid},), uhat_funcs)
ccall((:spir_basis_release, libpath), Cvoid, (Ptr{Cvoid},), basis)
kernel_release(kernel)

# Test 8: Omega (real frequency) sampling points
println("\n📝 Test 8: Omega Sampling Points")
kernel = kernel_logistic_new(10.0)

# Create basis
status_basis = Ref{Int32}(0)
basis = ccall(
    (:spir_basis_new, libpath),
    Ptr{Cvoid},
    (Int32, Float64, Float64, Float64, Ptr{Cvoid}, Ptr{Cvoid}, Int32, Ref{Int32}),
    1,      # Fermionic
    10.0,   # beta
    1.0,    # omega_max
    1e-6,   # epsilon
    kernel,
    C_NULL,
    -1,
    status_basis
)

if basis == C_NULL || status_basis[] != SPIR_COMPUTATION_SUCCESS
    error("Failed to create basis: status = $(status_basis[])")
end

# Get number of omega sampling points
n_ws = Ref{Int32}(0)
status_ws = ccall(
    (:spir_basis_get_n_default_ws, libpath),
    Int32,
    (Ptr{Cvoid}, Ref{Int32}),
    basis, n_ws
)

if status_ws != SPIR_COMPUTATION_SUCCESS
    error("Failed to get number of omega points: status = $status_ws")
end

println("✅ Number of omega sampling points: $(n_ws[])")

# Get omega sampling points
ws = Vector{Float64}(undef, n_ws[])
status_get_ws = ccall(
    (:spir_basis_get_default_ws, libpath),
    Int32,
    (Ptr{Cvoid}, Ptr{Float64}),
    basis, ws
)

if status_get_ws != SPIR_COMPUTATION_SUCCESS
    error("Failed to get omega points: status = $status_get_ws")
end

println("   First 5 omega points:")
for i in 1:min(5, length(ws))
    println("     ω[$i] = $(ws[i])")
end

# Test singular_values alias
basis_size = Ref{Int32}(0)
ccall(
    (:spir_basis_get_size, libpath),
    Int32,
    (Ptr{Cvoid}, Ref{Int32}),
    basis, basis_size
)

svals_alias = Vector{Float64}(undef, basis_size[])
status_alias = ccall(
    (:spir_basis_get_singular_values, libpath),
    Int32,
    (Ptr{Cvoid}, Ptr{Float64}),
    basis, svals_alias
)

if status_alias != SPIR_COMPUTATION_SUCCESS
    error("Failed to get singular values via alias: status = $status_alias")
end

println("✅ get_singular_values (alias) works correctly")
println("   First singular value: $(svals_alias[1])")

# Cleanup
ccall((:spir_basis_release, libpath), Cvoid, (Ptr{Cvoid},), basis)
kernel_release(kernel)

# Test 9: Function Evaluation
println("\n📝 Test 9: Function Evaluation")
kernel = kernel_logistic_new(10.0)

# Create basis
status_basis = Ref{Int32}(0)
basis = ccall(
    (:spir_basis_new, libpath),
    Ptr{Cvoid},
    (Int32, Float64, Float64, Float64, Ptr{Cvoid}, Ptr{Cvoid}, Int32, Ref{Int32}),
    1,      # Fermionic
    10.0,   # beta
    1.0,    # omega_max
    1e-6,   # epsilon
    kernel,
    C_NULL,
    -1,
    status_basis
)

# Get u funcs
status_u = Ref{Int32}(0)
u_funcs = ccall((:spir_basis_get_u, libpath), Ptr{Cvoid}, (Ptr{Cvoid}, Ref{Int32}), basis, status_u)

# Get size
u_size = Ref{Int32}(0)
ccall((:spir_funcs_get_size, libpath), Int32, (Ptr{Cvoid}, Ref{Int32}), u_funcs, u_size)

println("✅ u funcs size: $(u_size[])")

# Test single point evaluation
u_values = Vector{Float64}(undef, u_size[])
status_eval = ccall(
    (:spir_funcs_eval, libpath),
    Int32,
    (Ptr{Cvoid}, Float64, Ptr{Float64}),
    u_funcs, 0.0, u_values
)

if status_eval != SPIR_COMPUTATION_SUCCESS
    error("Failed to evaluate u: status = $status_eval")
end

println("✅ Evaluated u at tau=0")
println("   u[0](0) = $(u_values[1])")
println("   u[1](0) = $(u_values[2])")

# Test batch evaluation
tau_points = [-5.0, -2.5, 0.0, 2.5, 5.0]
u_batch = Matrix{Float64}(undef, u_size[], length(tau_points))  # column-major
status_batch = ccall(
    (:spir_funcs_batch_eval, libpath),
    Int32,
    (Ptr{Cvoid}, Int32, Int32, Ptr{Float64}, Ptr{Float64}),
    u_funcs, 1, length(tau_points), tau_points, u_batch
)

if status_batch != SPIR_COMPUTATION_SUCCESS
    error("Failed to batch evaluate u: status = $status_batch")
end

println("✅ Batch evaluated u at $(length(tau_points)) tau points")
println("   u[0](tau) = $(u_batch[1, :])")

# Get uhat funcs
status_uhat = Ref{Int32}(0)
uhat_funcs = ccall((:spir_basis_get_uhat, libpath), Ptr{Cvoid}, (Ptr{Cvoid}, Ref{Int32}), basis, status_uhat)

# Test Matsubara evaluation
uhat_values = Vector{ComplexF64}(undef, u_size[])
status_matsu = ccall(
    (:spir_funcs_eval_matsu, libpath),
    Int32,
    (Ptr{Cvoid}, Int64, Ptr{ComplexF64}),
    uhat_funcs, Int64(1), uhat_values
)

if status_matsu != SPIR_COMPUTATION_SUCCESS
    error("Failed to evaluate uhat: status = $status_matsu")
end

println("✅ Evaluated uhat at n=1 (iω_1)")
println("   uhat[0](iω_1) = $(uhat_values[1])")
println("   |uhat[0](iω_1)| = $(abs(uhat_values[1]))")

# Test batch Matsubara evaluation
matsu_ns = Int64[1, 3, 5, 7]
uhat_batch = Matrix{ComplexF64}(undef, u_size[], length(matsu_ns))
status_batch_matsu = ccall(
    (:spir_funcs_batch_eval_matsu, libpath),
    Int32,
    (Ptr{Cvoid}, Int32, Int32, Ptr{Int64}, Ptr{ComplexF64}),
    uhat_funcs, 1, length(matsu_ns), matsu_ns, uhat_batch
)

if status_batch_matsu != SPIR_COMPUTATION_SUCCESS
    error("Failed to batch evaluate uhat: status = $status_batch_matsu")
end

println("✅ Batch evaluated uhat at $(length(matsu_ns)) Matsubara frequencies")
println("   First column magnitudes: $(abs.(uhat_batch[1:3, 1]))")

# Cleanup
ccall((:spir_funcs_release, libpath), Cvoid, (Ptr{Cvoid},), u_funcs)
ccall((:spir_funcs_release, libpath), Cvoid, (Ptr{Cvoid},), uhat_funcs)
ccall((:spir_basis_release, libpath), Cvoid, (Ptr{Cvoid},), basis)
kernel_release(kernel)

# Test 10: Clone and Slice Functions
println("\n📝 Test 10: Clone and Slice Functions")
kernel = kernel_logistic_new(10.0)

# Create basis
status_basis = Ref{Int32}(0)
basis = ccall(
    (:spir_basis_new, libpath),
    Ptr{Cvoid},
    (Int32, Float64, Float64, Float64, Ptr{Cvoid}, Ptr{Cvoid}, Int32, Ref{Int32}),
    1,      # Fermionic
    10.0,   # beta
    1.0,    # omega_max
    1e-6,   # epsilon
    kernel,
    C_NULL,
    -1,
    status_basis
)

# Get u funcs
status_u = Ref{Int32}(0)
u_funcs = ccall((:spir_basis_get_u, libpath), Ptr{Cvoid}, (Ptr{Cvoid}, Ref{Int32}), basis, status_u)

# Test is_assigned
is_assigned = ccall((:spir_funcs_is_assigned, libpath), Int32, (Ptr{Cvoid},), u_funcs)
println("✅ is_assigned for valid object: $is_assigned")
@assert is_assigned == 1

is_null_assigned = ccall((:spir_funcs_is_assigned, libpath), Int32, (Ptr{Cvoid},), C_NULL)
println("✅ is_assigned for null: $is_null_assigned")
@assert is_null_assigned == 0

# Test clone
cloned_funcs = ccall((:spir_funcs_clone, libpath), Ptr{Cvoid}, (Ptr{Cvoid},), u_funcs)
@assert cloned_funcs != C_NULL
println("✅ Cloned funcs successfully")

# Check cloned size
u_size = Ref{Int32}(0)
ccall((:spir_funcs_get_size, libpath), Int32, (Ptr{Cvoid}, Ref{Int32}), u_funcs, u_size)
cloned_size = Ref{Int32}(0)
ccall((:spir_funcs_get_size, libpath), Int32, (Ptr{Cvoid}, Ref{Int32}), cloned_funcs, cloned_size)
@assert cloned_size[] == u_size[]
println("✅ Cloned funcs has same size ($(u_size[]))")

# Test slice
indices = Int32[0, 2, 4]  # Select first, third, fifth functions (0-indexed)
slice_status = Ref{Int32}(0)
sliced_funcs = ccall(
    (:spir_funcs_get_slice, libpath),
    Ptr{Cvoid},
    (Ptr{Cvoid}, Int32, Ptr{Int32}, Ref{Int32}),
    u_funcs,
    length(indices),
    indices,
    slice_status
)

@assert slice_status[] == SPIR_COMPUTATION_SUCCESS
@assert sliced_funcs != C_NULL
println("✅ Created slice with $(length(indices)) functions")

# Check sliced size
sliced_size = Ref{Int32}(0)
ccall((:spir_funcs_get_size, libpath), Int32, (Ptr{Cvoid}, Ref{Int32}), sliced_funcs, sliced_size)
@assert sliced_size[] == length(indices)
println("✅ Sliced funcs has correct size ($(sliced_size[]))")

# Evaluate sliced functions
sliced_values = Vector{Float64}(undef, sliced_size[])
status_eval = ccall(
    (:spir_funcs_eval, libpath),
    Int32,
    (Ptr{Cvoid}, Float64, Ptr{Float64}),
    sliced_funcs, 0.0, sliced_values
)
@assert status_eval == SPIR_COMPUTATION_SUCCESS
println("✅ Sliced funcs evaluates correctly")
println("   Values: $(sliced_values)")

# Test error case: negative index
bad_indices = Int32[-1]
bad_status = Ref{Int32}(0)
bad_slice = ccall(
    (:spir_funcs_get_slice, libpath),
    Ptr{Cvoid},
    (Ptr{Cvoid}, Int32, Ptr{Int32}, Ref{Int32}),
    u_funcs,
    length(bad_indices),
    bad_indices,
    bad_status
)
@assert bad_status[] == SPIR_INVALID_ARGUMENT
@assert bad_slice == C_NULL
println("✅ Negative index correctly rejected")

# Cleanup
ccall((:spir_funcs_release, libpath), Cvoid, (Ptr{Cvoid},), sliced_funcs)
ccall((:spir_funcs_release, libpath), Cvoid, (Ptr{Cvoid},), cloned_funcs)
ccall((:spir_funcs_release, libpath), Cvoid, (Ptr{Cvoid},), u_funcs)
ccall((:spir_basis_release, libpath), Cvoid, (Ptr{Cvoid},), basis)
kernel_release(kernel)

# Test 11: Extended Sampling Functions (_ext variants)
println("\n📝 Test 11: Extended Sampling Functions (_ext)")
kernel = kernel_logistic_new(10.0)

# Create basis
status_basis = Ref{Int32}(0)
basis = ccall(
    (:spir_basis_new, libpath),
    Ptr{Cvoid},
    (Int32, Float64, Float64, Float64, Ptr{Cvoid}, Ptr{Cvoid}, Int32, Ref{Int32}),
    1,      # Fermionic
    10.0,   # beta
    1.0,    # omega_max
    1e-6,   # epsilon
    kernel,
    C_NULL,
    -1,
    status_basis
)

# Test get_default_taus_ext - request 5 points
requested_tau = 5
tau_points_ext = Vector{Float64}(undef, requested_tau)
tau_returned = Ref{Int32}(0)
status_tau_ext = ccall(
    (:spir_basis_get_default_taus_ext, libpath),
    Int32,
    (Ptr{Cvoid}, Int32, Ptr{Float64}, Ref{Int32}),
    basis, Int32(requested_tau), tau_points_ext, tau_returned
)

@assert status_tau_ext == SPIR_COMPUTATION_SUCCESS
@assert tau_returned[] == requested_tau
println("✅ get_default_taus_ext returned $(tau_returned[]) points (requested $requested_tau)")
println("   First 3: $(tau_points_ext[1:3])")

# Test get_n_default_matsus_ext - request 3 points
requested_matsu = 3
matsu_count_ext = Ref{Int32}(0)
status_matsu_count_ext = ccall(
    (:spir_basis_get_n_default_matsus_ext, libpath),
    Int32,
    (Ptr{Cvoid}, Bool, Int32, Ref{Int32}),
    basis, true, Int32(requested_matsu), matsu_count_ext
)

@assert status_matsu_count_ext == SPIR_COMPUTATION_SUCCESS
@assert matsu_count_ext[] == requested_matsu
println("✅ get_n_default_matsus_ext returned count: $(matsu_count_ext[])")

# Test get_default_matsus_ext
matsu_points_ext = Vector{Int64}(undef, matsu_count_ext[])
matsu_returned = Ref{Int32}(0)
status_matsu_ext = ccall(
    (:spir_basis_get_default_matsus_ext, libpath),
    Int32,
    (Ptr{Cvoid}, Bool, Int32, Ptr{Int64}, Ref{Int32}),
    basis, true, Int32(requested_matsu), matsu_points_ext, matsu_returned
)

@assert status_matsu_ext == SPIR_COMPUTATION_SUCCESS
@assert matsu_returned[] == matsu_count_ext[]
println("✅ get_default_matsus_ext returned $(matsu_returned[]) matsubara points")
println("   Points: $(matsu_points_ext)")

# Test requesting more points than available
big_request = 100
big_tau_points = Vector{Float64}(undef, big_request)
big_tau_returned = Ref{Int32}(0)
status_big = ccall(
    (:spir_basis_get_default_taus_ext, libpath),
    Int32,
    (Ptr{Cvoid}, Int32, Ptr{Float64}, Ref{Int32}),
    basis, Int32(big_request), big_tau_points, big_tau_returned
)

@assert status_big == SPIR_COMPUTATION_SUCCESS
println("✅ Requesting 100 points returned $(big_tau_returned[]) actual points")
println("   (Returns min(requested, available))")

# Cleanup
ccall((:spir_basis_release, libpath), Cvoid, (Ptr{Cvoid},), basis)
kernel_release(kernel)

println("\n" * "=" ^ 50)
println("Test 12: Sampling API - Tau Sampling")
println("=" ^ 50)

# Create a small basis for testing
kernel = kernel_logistic_new(10.0)
sve_status = Ref{Cint}(0)
sve = ccall((:spir_sve_result_new, libpath), Ptr{Cvoid},
    (Ptr{Cvoid}, Cdouble, Cdouble, Cint, Cint, Cint, Ptr{Cint}),
    kernel, 1e-6, -1.0, -1, -1, -1, sve_status)
@assert sve_status[] == SPIR_COMPUTATION_SUCCESS

basis_status = Ref{Cint}(0)
basis = ccall((:spir_basis_new, libpath), Ptr{Cvoid},
    (Cint, Cdouble, Cdouble, Cdouble, Ptr{Cvoid}, Ptr{Cvoid}, Cint, Ptr{Cint}),
    1, 10.0, 1.0, 1e-6, kernel, sve, 5, basis_status)  # max_size=5
@assert basis_status[] == SPIR_COMPUTATION_SUCCESS

# Get basis size
basis_size_ref = Ref{Cint}(0)
status = ccall((:spir_basis_get_size, libpath), Cint,
    (Ptr{Cvoid}, Ptr{Cint}), basis, basis_size_ref)
@assert status == SPIR_COMPUTATION_SUCCESS
basis_size = basis_size_ref[]
println("📊 Basis size: $basis_size")

# Create tau sampling points
tau_points = [10.0 * (i + 1) / (basis_size + 1) for i in 0:basis_size-1]
println("📍 Tau sampling points: $tau_points")

# Create tau sampling
status_ref = Ref{Cint}(0)
sampling = ccall((:spir_tau_sampling_new, libpath), Ptr{Cvoid},
    (Ptr{Cvoid}, Cint, Ptr{Cdouble}, Ptr{Cint}),
    basis, length(tau_points), tau_points, status_ref)
@assert status_ref[] == SPIR_COMPUTATION_SUCCESS
@assert sampling != C_NULL
println("✅ Tau sampling created")

# Get number of sampling points
n_points_ref = Ref{Cint}(0)
status = ccall((:spir_sampling_get_npoints, libpath), Cint,
    (Ptr{Cvoid}, Ptr{Cint}), sampling, n_points_ref)
@assert status == SPIR_COMPUTATION_SUCCESS
@assert n_points_ref[] == basis_size
println("✅ Number of points: $(n_points_ref[])")

# Get tau points back
retrieved_taus = zeros(Float64, basis_size)
status = ccall((:spir_sampling_get_taus, libpath), Cint,
    (Ptr{Cvoid}, Ptr{Cdouble}), sampling, retrieved_taus)
@assert status == SPIR_COMPUTATION_SUCCESS
@assert maximum(abs.(retrieved_taus .- tau_points)) < 1e-10
println("✅ Retrieved tau points match")

# Test evaluate: coefficients → values at sampling points
# Create simple test coefficients
coeffs = ones(Float64, basis_size)
values = zeros(Float64, basis_size)

# 1D array: ndim=1, dims=[basis_size], target_dim=0
dims_1d = Cint[basis_size]
status = ccall((:spir_sampling_eval_dd, libpath), Cint,
    (Ptr{Cvoid}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}, Ptr{Cdouble}),
    sampling, 1, 1, dims_1d, 0, coeffs, values)  # order=1 (column-major, Julia default)
@assert status == SPIR_COMPUTATION_SUCCESS
println("✅ Eval 1D: $(values[1:min(3,end)]...)")

# Test fit: values → coefficients
fitted_coeffs = zeros(Float64, basis_size)
status = ccall((:spir_sampling_fit_dd, libpath), Cint,
    (Ptr{Cvoid}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}, Ptr{Cdouble}),
    sampling, 1, 1, dims_1d, 0, values, fitted_coeffs)
@assert status == SPIR_COMPUTATION_SUCCESS
println("✅ Fit 1D: roundtrip error = $(maximum(abs.(fitted_coeffs .- coeffs)))")

# Test 2D array (batch processing)
batch_size = 3
coeffs_2d = ones(Float64, basis_size, batch_size)
values_2d_result = zeros(Float64, basis_size, batch_size)
dims_2d = Cint[basis_size, batch_size]

status = ccall((:spir_sampling_eval_dd, libpath), Cint,
    (Ptr{Cvoid}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}, Ptr{Cdouble}),
    sampling, 1, 2, dims_2d, 0, coeffs_2d, values_2d_result)
@assert status == SPIR_COMPUTATION_SUCCESS
println("✅ Eval 2D: shape $(size(values_2d_result))")

# Cleanup sampling
ccall((:spir_sampling_release, libpath), Cvoid, (Ptr{Cvoid},), sampling)
println("✅ Sampling released")

println("\n" * "=" ^ 50)
println("Test 13: Sampling API - Matsubara Sampling")
println("=" ^ 50)

# Get default Matsubara sampling points
n_matsu_ref = Ref{Cint}(0)
status = ccall((:spir_basis_get_n_default_matsus, libpath), Cint,
    (Ptr{Cvoid}, Bool, Ptr{Cint}), basis, false, n_matsu_ref)
@assert status == SPIR_COMPUTATION_SUCCESS
n_matsu = n_matsu_ref[]
println("📊 Number of default Matsubara points: $n_matsu")

matsu_points = zeros(Int64, n_matsu)
status = ccall((:spir_basis_get_default_matsus, libpath), Cint,
    (Ptr{Cvoid}, Bool, Ptr{Int64}), basis, false, matsu_points)
@assert status == SPIR_COMPUTATION_SUCCESS
println("📍 Matsubara points: $(matsu_points[1:min(5,end)]...)")

# Create Matsubara sampling (positive_only=false)
status_ref = Ref{Cint}(0)
matsu_sampling = ccall((:spir_matsu_sampling_new, libpath), Ptr{Cvoid},
    (Ptr{Cvoid}, Bool, Cint, Ptr{Int64}, Ptr{Cint}),
    basis, false, length(matsu_points), matsu_points, status_ref)
@assert status_ref[] == SPIR_COMPUTATION_SUCCESS
@assert matsu_sampling != C_NULL
println("✅ Matsubara sampling created (full range)")

# Get matsus back
retrieved_matsus = zeros(Int64, n_matsu)
status = ccall((:spir_sampling_get_matsus, libpath), Cint,
    (Ptr{Cvoid}, Ptr{Int64}), matsu_sampling, retrieved_matsus)
@assert status == SPIR_COMPUTATION_SUCCESS
@assert all(retrieved_matsus .== matsu_points)
println("✅ Retrieved Matsubara points match")

# Test eval_zz: complex coefficients → complex values
coeffs_complex = ones(ComplexF64, basis_size)
values_complex = zeros(ComplexF64, n_matsu)
dims_matsu = Cint[basis_size]

status = ccall((:spir_sampling_eval_zz, libpath), Cint,
    (Ptr{Cvoid}, Cint, Cint, Ptr{Cint}, Cint, Ptr{ComplexF64}, Ptr{ComplexF64}),
    matsu_sampling, 1, 1, dims_matsu, 0, coeffs_complex, values_complex)
@assert status == SPIR_COMPUTATION_SUCCESS
println("✅ Eval complex 1D: $(abs.(values_complex[1:min(3,end)])...)")

# Test fit_zz: complex values → complex coefficients
fitted_complex = zeros(ComplexF64, basis_size)
dims_matsu_fit = Cint[n_matsu]
status = ccall((:spir_sampling_fit_zz, libpath), Cint,
    (Ptr{Cvoid}, Cint, Cint, Ptr{Cint}, Cint, Ptr{ComplexF64}, Ptr{ComplexF64}),
    matsu_sampling, 1, 1, dims_matsu_fit, 0, values_complex, fitted_complex)
@assert status == SPIR_COMPUTATION_SUCCESS
println("✅ Fit complex 1D: roundtrip error = $(maximum(abs.(fitted_complex .- coeffs_complex)))")

# Cleanup
ccall((:spir_sampling_release, libpath), Cvoid, (Ptr{Cvoid},), matsu_sampling)
ccall((:spir_basis_release, libpath), Cvoid, (Ptr{Cvoid},), basis)
ccall((:spir_sve_result_release, libpath), Cvoid, (Ptr{Cvoid},), sve)
kernel_release(kernel)

println("\n" * "=" ^ 50)
println("Test 14: DLR API")
println("=" ^ 50)

# Re-create kernel and basis for DLR test
kernel_status = Ref{Cint}(0)
kernel = ccall((:spir_logistic_kernel_new, libpath), Ptr{Cvoid},
    (Cdouble, Ptr{Cint}), 10.0, kernel_status)
@assert kernel_status[] == SPIR_COMPUTATION_SUCCESS
@assert kernel != C_NULL

sve_status = Ref{Cint}(0)
sve = ccall((:spir_sve_result_new, libpath), Ptr{Cvoid},
    (Ptr{Cvoid}, Cdouble, Cdouble, Cint, Cint, Cint, Ptr{Cint}),
    kernel, 1e-6, 1e-6, -1, -1, 0, sve_status)
@assert sve_status[] == SPIR_COMPUTATION_SUCCESS

basis_status = Ref{Cint}(0)
ir_basis = ccall((:spir_basis_new, libpath), Ptr{Cvoid},
    (Cint, Cdouble, Cdouble, Cdouble, Ptr{Cvoid}, Ptr{Cvoid}, Cint, Ptr{Cint}),
    1, 10.0, 1.0, 1e-6, kernel, sve, -1, basis_status)  # 1 = Fermionic
@assert basis_status[] == SPIR_COMPUTATION_SUCCESS
@assert ir_basis != C_NULL

# Create DLR from IR basis
dlr_status = Ref{Cint}(0)
dlr = ccall((:spir_dlr_new, libpath), Ptr{Cvoid},
    (Ptr{Cvoid}, Ptr{Cint}), ir_basis, dlr_status)
@assert dlr_status[] == SPIR_COMPUTATION_SUCCESS
@assert dlr != C_NULL
println("✅ DLR created from IR basis")

# Get number of poles
npoles_ref = Ref{Cint}(0)
status = ccall((:spir_dlr_get_npoles, libpath), Cint,
    (Ptr{Cvoid}, Ptr{Cint}), dlr, npoles_ref)
@assert status == SPIR_COMPUTATION_SUCCESS
npoles = npoles_ref[]
println("📊 DLR has $npoles poles")

# Get poles
poles = zeros(Float64, npoles)
status = ccall((:spir_dlr_get_poles, libpath), Cint,
    (Ptr{Cvoid}, Ptr{Cdouble}), dlr, poles)
@assert status == SPIR_COMPUTATION_SUCCESS
println("📍 First 3 poles: $(poles[1:min(3,end)]...)")

# Get u funcs from DLR
u_status = Ref{Cint}(0)
u_funcs = ccall((:spir_basis_get_u, libpath), Ptr{Cvoid},
    (Ptr{Cvoid}, Ptr{Cint}), dlr, u_status)
@assert u_status[] == SPIR_COMPUTATION_SUCCESS
@assert u_funcs != C_NULL
println("✅ Got u funcs from DLR")

# Evaluate u at tau=0.5
tau = 0.5
u_values = zeros(Float64, npoles)
status = ccall((:spir_funcs_eval, libpath), Cint,
    (Ptr{Cvoid}, Cdouble, Ptr{Cdouble}), u_funcs, tau, u_values)
@assert status == SPIR_COMPUTATION_SUCCESS
println("✅ Evaluated u at τ=$tau: $(u_values[1:min(3,end)]...)")

# Get uhat funcs from DLR
uhat_status = Ref{Cint}(0)
uhat_funcs = ccall((:spir_basis_get_uhat, libpath), Ptr{Cvoid},
    (Ptr{Cvoid}, Ptr{Cint}), dlr, uhat_status)
@assert uhat_status[] == SPIR_COMPUTATION_SUCCESS
@assert uhat_funcs != C_NULL
println("✅ Got uhat funcs from DLR")

# Evaluate uhat at Matsubara frequency n=1
n_matsu = 1
uhat_values = zeros(ComplexF64, npoles)
status = ccall((:spir_funcs_eval_matsu, libpath), Cint,
    (Ptr{Cvoid}, Int64, Ptr{ComplexF64}), uhat_funcs, n_matsu, uhat_values)
@assert status == SPIR_COMPUTATION_SUCCESS
println("✅ Evaluated uhat at n=$n_matsu: |uhat|=$(abs.(uhat_values[1:min(3,end)])...)")

# Cleanup
ccall((:spir_funcs_release, libpath), Cvoid, (Ptr{Cvoid},), uhat_funcs)
ccall((:spir_funcs_release, libpath), Cvoid, (Ptr{Cvoid},), u_funcs)
ccall((:spir_basis_release, libpath), Cvoid, (Ptr{Cvoid},), dlr)
ccall((:spir_basis_release, libpath), Cvoid, (Ptr{Cvoid},), ir_basis)
ccall((:spir_sve_result_release, libpath), Cvoid, (Ptr{Cvoid},), sve)
ccall((:spir_kernel_release, libpath), Cvoid, (Ptr{Cvoid},), kernel)

println("\n" * "=" ^ 50)
println("✅ All tests passed (including Sampling API and DLR)!")

