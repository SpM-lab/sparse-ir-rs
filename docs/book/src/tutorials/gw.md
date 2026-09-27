# GW

*Ported from the Python notebook `GW_py.ipynb` of sparse-ir-tutorial. The
program that produced every number and figure on this page is
`docs/tutorial-code/src/bin/gw.rs`.*

The previous page evaluated one diagram once. This one runs a self-consistent
loop, and in doing so touches every representation the library offers —
several times per iteration, in both statistics.

## Theory

The Hedin equations, in the \\(GW\\) approximation and for a single site with
a bare interaction \\(U\\), are

\\[
P(\tau) = G(\tau)\, G(\beta - \tau),
\qquad
W(\mathrm{i}\omega) = \frac{U}{1 - U P(\mathrm{i}\omega)},
\qquad
\Sigma(\tau) = G(\tau)\, W(\tau),
\\]

closed by the Dyson equation
\\(G^{-1} = G_0^{-1} - \Sigma\\). Each of them is diagonal in exactly one
representation: the two products live in imaginary time, the screening is
algebraic on the Matsubara axis, and so is the Dyson equation. One iteration
is therefore a tour:

```text
G(iν) → Gₗ → G(τ) → P(τ) → Pₗ → P(iω) → W(iω) → Wₗ → W(τ) → Σ(τ) → Σₗ → Σ(iν) → G(iν)
```

Only the constant part of \\(W\\) is split off and left behind: what is carried
into \\(\Sigma\\) is \\(U/(1 - UP) - U\\), the bare interaction itself being
the first-order (Hartree) term, which is computed separately as
\\(U G(\beta^-)\\) and kept out of the Dyson equation — it is a shift of the
chemical potential.

## Two statistics, one loop

\\(G\\) and \\(\Sigma\\) are fermionic; \\(P\\) and \\(W\\) are bosonic. The
two products mix them: \\(P\\) is a product of \\(G\\)'s evaluated at the
*bosonic* sampling times, and \\(\Sigma\\) a product of \\(W\\)'s at the
*fermionic* ones. So the example carries two bases and two meshes:

```rust,ignore
let basis_f = FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel_f, BETA, None, None)?;
let basis_b = FiniteTempBasis::<LogisticKernel, Bosonic>::new(kernel_b, BETA, None, None)?;
let mesh_f = IrMesh::<Fermionic>::new(&basis_f)?;
let mesh_b = IrMesh::<Bosonic>::new(&basis_b)?;
```

and two evaluation matrices, built once outside the loop:

```rust,ignore
// The fermionic basis functions at the bosonic sampling times, and vice versa.
let uf_at_tau_b = basis_f.evaluate_tau(mesh_b.tau_points())?;
let ub_at_tau_f = basis_b.evaluate_tau(mesh_f.tau_points())?;
```

`Basis::evaluate_tau` is the operation that makes this safe. The sampling
times live on \\([-\beta/2, \beta/2]\\), so evaluating a basis function there
needs the (anti-)periodicity relation — and *which* relation is a property of
the function, never of the grid it is being sampled on. \\(G\\) stays
anti-periodic when evaluated at bosonic times; \\(W\\) stays periodic at
fermionic ones. `evaluate_tau` applies the statistics of the basis it belongs
to, which is exactly right. Doing the same thing by hand — folding the times
and then multiplying by a sign taken from the *grid* — is the mistake this
example exists to rule out.

![G at the sampling times](gw_green_tau.png)

## The reversal, again

\\(P(\tau) = G(\tau) G(\beta - \tau)\\) needs the same reversal as the
[second-order self-energy](second_order_perturbation.md), and here it has a
wrinkle. At \\(\beta = 10\\), \\(\omega_\mathrm{max} = 1\\) and the default
accuracy, both grids hold 19 points: 18 of them in \\(\pm\\) pairs, and one at
exactly \\(\tau = \beta/2\\). That last point has no partner under
\\(\tau \to -\tau\\) on the grid — its mirror \\(-\beta/2\\) is the same point
one period away. The reversal must therefore allow for a wrap, which brings a
second \\(\zeta\\); at \\(\beta/2\\) the two cancel and the row maps to itself
with a plus sign.

```rust,ignore
// G is fermionic even though these are the bosonic times.
let g_beta_minus_tau = mesh_b.reverse_tau_as::<Fermionic>(&g_tau_b, 1);
let p_tau_b: Vec<Complex64> = g_tau_b
    .iter()
    .zip(&g_beta_minus_tau)
    .map(|(g, g_reversed)| g * g_reversed)
    .collect();
```

For this particular model the check is worth stating plainly: the atom is
particle-hole symmetric, so \\(G(\beta - \tau) = G(\tau)\\) and a plot of the
two would show one curve. The fermionic and bosonic sampling times happen to
coincide as well. Neither the picture nor the abscissa would reveal a missing
\\(\zeta\\) — only the numbers would, which is why the relation is pinned by
`docs/tutorial-code/tests/tau_convention.rs` and by a comparison of every
table on this page against the Python implementation.

![P at the sampling times](gw_polarization_tau.png)

## The screened interaction

\\(W\\) is algebraic on the Matsubara axis and then has to come back to
imaginary time — on the *fermionic* grid, because that is where it meets
\\(G\\):

```rust,ignore
let w_iw_b: Vec<Complex64> = p_iw_b.iter().map(|p| U / (1.0 - U * p) - U).collect();
let w_l_b = mesh_b.wn_to_l(&w_iw_b, 1)?;
let w_tau_f = contract(&ub_at_tau_f, &w_l_b);   // bosonic basis, fermionic times
```

![W on both axes](gw_screened.png)

## The self-energy, and the loop

```rust,ignore
let e_tau_f: Vec<Complex64> = g_tau_f.iter().zip(&w_tau_f).map(|(g, w)| g * w).collect();
let e_l_f = mesh_f.tau_to_l(&e_tau_f, 1)?;
let e_iw_f = mesh_f.l_to_wn(&e_l_f, 1)?;
let hartree: Complex64 = U * contract(&uf_at_beta, &g_l_f)[0];
// The Dyson equation, with the Hartree term left out of it.
g_iw_f = ...  // 1/(1/G₀ − Σ)
```

![Σ at the sampling times](gw_self_energy_tau.png)

Both expansions fall off like the singular values of their own basis, which is
the statement that a product of Green's functions is still a function the
basis represents well — the fact the whole method rests on:

![The IR coefficients of P and Σ](gw_coefficients.png)

Twenty iterations are more than enough: the change in \\(\Sigma\\) falls by
about a decade per step and reaches the rounding floor around iteration 18.

![Convergence](gw_convergence.png)

![Σ at the fixed point](gw_self_energy_matsubara.png)

![G before and after](gw_green_matsubara.png)

## Key API pieces

| What you want | What to call |
| --- | --- |
| a basis function of one statistics at the other's sampling times | `Basis::evaluate_tau` |
| \\(u_l(\beta^-)\\), for the Hartree term | `Basis::evaluate_tau(&[beta])` |
| \\(G(\beta - \tau)\\) for a function whose statistics is not the mesh's | `IrMesh::reverse_tau_as::<S>` |
| the round trip through the basis | `IrMesh::wn_to_l`, `l_to_tau`, `tau_to_l`, `l_to_wn` |

The whole loop is 19 + 19 coefficients wide. Nothing in it ever sees a dense
imaginary-time grid.
