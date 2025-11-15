# Verify DLR evaluation for single pole case
# Test case: Bosonic statistics, single pole at npoles/2

# Parameters from test
beta = 1e+3
wmax = 2.0
pole_idx = 19  # npoles/2 for 38 poles
pole = 0.00121101  # From debug output: poles[npoles/2]

# Tau points (from test output)
tau_points = [
    -445.386, -342.248, -254.412, -185.105, -133.007,
    -94.917, -67.4781, -47.8588, -33.8764, -23.9175,
    -16.8128, -11.7241, -8.05622, -5.39396, -3.45491,
    -2.05261, -1.06726, -0.424217, -0.0794999,
    0.0794999, 0.424217, 1.06726, 2.05261, 3.45491,
    5.39396, 8.05622, 11.7241, 16.8128, 23.9175,
    33.8764, 47.8588, 67.4781, 94.917, 133.007,
    185.105, 254.412, 342.248, 445.386
]

# Normalize tau function for Bosonic statistics
function normalize_tau_bosonic(tau, beta)
    if tau < -beta || tau > beta
        error("tau = $tau is outside allowed range [-beta = $beta, beta = $beta]")
    end
    
    # Special handling for negative zero
    if tau == 0.0 && signbit(tau)
        return (beta, 1.0)  # Periodic: wraps to beta with sign unchanged
    end
    
    # If already in [0, β], return as-is with sign = 1
    if tau >= 0.0 && tau <= beta
        return (tau, 1.0)
    end
    
    # tau ∈ [-β, 0): wrap to [0, β]
    tau_normalized = tau + beta
    sign = 1.0  # Bosonic: periodic, sign stays
    
    (tau_normalized, sign)
end

# Logistic kernel computation
# K(x, y) = exp(-lambda * y * (x + 1) / 2) / (1 + exp(-lambda * y))
function logistic_kernel(x, y, lambda)
    exp_arg = -lambda * y * (x + 1) / 2
    numerator = exp(exp_arg)
    denominator = 1 + exp(-lambda * y)
    numerator / denominator
end

# Compute inv_weight for Bosonic
inv_weight = tanh(beta * pole / 2)

# Lambda for kernel
lambda = beta * wmax

# Compute DLR basis function value for each tau
println("Computing DLR values for single pole case:")
println("beta = $beta, wmax = $wmax, pole = $pole")
println("inv_weight = $inv_weight")
println()

gtau_dlr = Float64[]

for tau in tau_points
    # Normalize tau
    tau_norm, sign = normalize_tau_bosonic(tau, beta)
    
    # Compute kernel parameters
    x = 2.0 * tau_norm / beta - 1.0
    y = pole / wmax
    
    # Compute kernel value
    kernel_val = logistic_kernel(x, y, lambda)
    
    # DLR basis function: u_i(τ) = sign * (-K(x, y_i)) * inv_weight[i]
    u_value = sign * (-kernel_val) * inv_weight
    
    # Green's function: G(τ) = coeffs[i] * u_i(τ) = 1.0 * u_value (single pole)
    gtau = 1.0 * u_value
    
    push!(gtau_dlr, gtau)
    
    # Print first few and last few
    idx = findfirst(==(tau), tau_points)
    if idx <= 5 || idx >= length(tau_points) - 4
        println("tau=$tau, tau_norm=$tau_norm, sign=$sign, x=$x, y=$y, kernel_val=$kernel_val, u_value=$u_value, gtau=$gtau")
    end
end

println()
println("gtau_from_DLR (computed):")
for (i, val) in enumerate(gtau_dlr)
    if i <= 5 || i >= length(gtau_dlr) - 4
        println("  $val")
    end
end

