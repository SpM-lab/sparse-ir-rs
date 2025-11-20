### A Pluto.jl notebook ###
# v0.20.19

using Markdown
using InteractiveUtils

# ╔═╡ 241d9650-bf6c-11f0-a388-39fe8a039b7e
begin
	using SparseIR
	using Libdl: dlext
	# Load the shared library"
	const libpath = "../../target/debug/libsparse_ir_capi.$(dlext)"
end

# ╔═╡ 1758e5e0-1908-4b30-8e86-329f979b1e0e
pkgversion(SparseIR)

# ╔═╡ 6c6f6202-02a9-416b-ba6b-4f718a42bd5d
begin
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
	
	kernel = kernel_logistic_new(10.0)
end

# ╔═╡ 25d100cd-bcc1-446e-a14d-79e1ee8ccb33
begin
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
end

# ╔═╡ 2b548ca0-069f-478d-90b4-ad16e93497b8
begin
	status_v = Ref{Int32}(0)
	v_funcs = ccall(
	    (:spir_basis_get_v, libpath),
	    Ptr{Cvoid},
	    (Ptr{Cvoid}, Ref{Int32}),
	    basis, status_v
	)
end

# ╔═╡ 76f6ffb2-69ef-4c84-8dac-3e92e570b3fa
let
	status_uhat = Ref{Int32}(0)
	uhat_funcs = ccall(
	    (:spir_basis_get_uhat, libpath),
	    Ptr{Cvoid},
	    (Ptr{Cvoid}, Ptr{Int32}),
	    basis, status_uhat
	)
	println(status_uhat[])

	uhat_size = Ref{Int32}(0)
	ccall(
		(:spir_funcs_get_size, libpath),
		Cint,
		(Ptr{Cvoid}, Ptr{Cint}),
		uhat_funcs, uhat_size
	)
	uhat_size[]
end

# ╔═╡ bf89ff80-c6ab-4375-901d-3714229c6a60
let
	status_uhat_full = Ref{Int32}(0)
	uhat_full_funcs = ccall(
	    (:spir_basis_get_uhat_full, libpath),
	    Ptr{Cvoid},
	    (Ptr{Cvoid}, Ptr{Int32}),
	    basis, status_uhat_full
	)
	println(status_uhat_full[])

	uhat_full_size = Ref{Int32}(0)
	ccall(
		(:spir_funcs_get_size, libpath),
		Cint,
		(Ptr{Cvoid}, Ptr{Cint}),
		uhat_full_funcs, uhat_full_size
	)
	uhat_full_size[]
end

# ╔═╡ 91eeba52-05fc-4faf-b6c1-eb3d297a0e2b
length(FiniteTempBasis{Fermionic}(10.0, 1.0, 1e-6, kernel=LogisticKernel(10.)).uhat)

# ╔═╡ 071e7e32-c8f8-4046-9547-ac33cc475c7e
length(FiniteTempBasis{Fermionic}(10.0, 1.0, 1e-6, kernel=LogisticKernel(10.)).uhat_full)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Libdl = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
SparseIR = "4fe2279e-80f0-4adb-8463-ee114ff56b7d"

[compat]
SparseIR = "~1.1.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.1"
manifest_format = "2.0"
project_hash = "fd19f7878d1cb7e90ceefcc4e970689f20c76395"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Bessels]]
git-tree-sha1 = "4435559dc39793d53a9e3d278e185e920b4619ef"
uuid = "0e736298-9ec6-45e8-9647-e4fc86a2fe38"
version = "0.2.8"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "6c72198e6a101cccdd4c9731d3985e904ba26037"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.GenericLinearAlgebra]]
deps = ["LinearAlgebra", "Printf", "Random", "libblastrampoline_jll"]
git-tree-sha1 = "ad599869948d79efd63a030c970e2c6e21fecf4a"
uuid = "14197337-ba66-59df-a3e3-ca00e7dcff7a"
version = "0.3.17"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.MultiFloats]]
deps = ["LinearAlgebra", "Printf", "Random", "SIMD"]
git-tree-sha1 = "39ffa6286f40544ecea725d8031c615e79d88d45"
uuid = "bdf0d083-296b-4888-a5b6-7498122e68a5"
version = "2.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "fea870727142270bdf7624ad675901a1ee3b4c87"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.1"

[[deps.SparseIR]]
deps = ["Bessels", "GenericLinearAlgebra", "LinearAlgebra", "MultiFloats", "PrecompileTools", "QuadGK"]
git-tree-sha1 = "175036112d3e0ff85e967ae1919b33fefa0cb637"
uuid = "4fe2279e-80f0-4adb-8463-ee114ff56b7d"
version = "1.1.4"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"
"""

# ╔═╡ Cell order:
# ╠═241d9650-bf6c-11f0-a388-39fe8a039b7e
# ╠═1758e5e0-1908-4b30-8e86-329f979b1e0e
# ╠═25d100cd-bcc1-446e-a14d-79e1ee8ccb33
# ╠═6c6f6202-02a9-416b-ba6b-4f718a42bd5d
# ╠═2b548ca0-069f-478d-90b4-ad16e93497b8
# ╠═76f6ffb2-69ef-4c84-8dac-3e92e570b3fa
# ╠═bf89ff80-c6ab-4375-901d-3714229c6a60
# ╠═91eeba52-05fc-4faf-b6c1-eb3d297a0e2b
# ╠═071e7e32-c8f8-4046-9547-ac33cc475c7e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
