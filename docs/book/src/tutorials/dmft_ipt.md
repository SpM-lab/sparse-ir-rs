# DMFT with an IPT solver

*Ported from the Python notebook
[`DMFT_IPT_py.ipynb`](https://spm-lab.github.io/sparse-ir-tutorial-v2/src/DMFT_IPT_py.html)
of the [sparse-ir tutorials](https://spm-lab.github.io/sparse-ir-tutorial-v2/),
whose author is Niklas Witt. The programs that produced every number and
figure on this page are `docs/tutorial-code/src/bin/dmft_ipt.rs` and
`docs/tutorial-code/src/bin/dmft_ipt_scan.rs`; the loop itself is in
`docs/tutorial-code/src/dmft.rs`, and the code below is included from there.*

Every applied page so far computed something once. This one iterates: a
self-consistency loop that goes through the basis twice per step, several
thousand times over. That makes it the page where the basis has to be cheap,
and the page where "has it converged?" turns out to be the hard question.

## The model and the loop

Dynamical mean-field theory replaces a lattice problem by a single impurity
in a bath that is fixed by demanding that the impurity's Green's function is
the lattice's local one. On the Bethe lattice, with the semicircular density
of states

\\[
\rho(\omega) = \frac{2}{\pi D^2}\sqrt{D^2 - \omega^2},
\qquad D = 2t_\star = 2,
\\]

the self-consistency condition collapses to one line,

\\[
\mathcal{G}^{-1}(\mathrm{i}\nu) = \mathrm{i}\nu - t_\star^2 G_\mathrm{loc}(\mathrm{i}\nu),
\\]

and iterated perturbation theory approximates the impurity self-energy by the
second-order diagram, which at half filling is just a cube in imaginary time:

\\[
\Sigma(\tau) = U^2 \mathcal{G}(\tau)^3,
\qquad
G_\mathrm{loc}^{-1}(\mathrm{i}\nu) = \mathcal{G}^{-1}(\mathrm{i}\nu) - \Sigma(\mathrm{i}\nu).
\\]

\\(\Sigma\\) is local in \\(\tau\\) and the Dyson equation is local in
\\(\mathrm{i}\nu\\), so each iteration is a round trip between the two
representations — exactly what the basis is for:

```rust,ignore
{{#include ../../../tutorial-code/src/dmft.rs:ipt_step}}
```

At \\(\beta = 20\\), \\(\omega_\mathrm{max} = 2D = 4\\) and
\\(\varepsilon = 10^{-15}\\) the basis has 37 functions, with 37 sampling
times and 38 sampling frequencies. That is the entire state of the
calculation: 37 coefficients carry a function whose Matsubara tail reaches
out to \\(\nu \sim 10^2\\), and the round trip is two small dense solves.
Nothing in the loop grows with \\(\beta\\) except logarithmically, which is
what makes a 5000-iteration run a matter of seconds.

The non-interacting starting point comes from the same basis. The spectral
representation \\(G^0(\mathrm{i}\nu) = \int\mathrm{d}\omega\,
\rho(\omega)/(\mathrm{i}\nu - \omega)\\) becomes, in IR coefficients,
\\(g_l = -s_l \rho_l\\) with \\(\rho_l = \int\mathrm{d}\omega\,
v_l(\omega)\rho(\omega)\\). The overlap \\(\rho_l\\) is computed by
Gauss–Legendre quadrature after the substitution
\\(\omega = D\sin\theta\\), which removes the square-root singularity at
the band edges, so the result is exact to machine precision:

```rust,ignore
{{#include ../../../tutorial-code/src/dmft.rs:noninteracting}}
```

The mixing is the notebook's: \\(\Sigma \leftarrow 0.25\,\Sigma_\mathrm{new}
+ 0.75\,\Sigma_\mathrm{old}\\).

## One solve at \\(U = 5\\)

Run the loop at \\(U = 5\\) with the notebook's stopping rule — stop when the
relative change of \\(\Sigma\\) falls below \\(10^{-5}\\) — and it stops at
iteration 65, reporting

\\[
Z = \left(1 - \frac{\partial\,\mathrm{Im}\,\Sigma}{\partial \nu}\right)^{-1}
  \approx 0.263,
\\]

a quasiparticle weight of a quarter: a correlated metal. The derivative is a
finite difference over the two lowest positive frequencies,
\\(\partial\,\mathrm{Im}\,\Sigma/\partial\nu \approx
[\mathrm{Im}\,\Sigma(\mathrm{i}\nu_3) - \mathrm{Im}\,\Sigma(\mathrm{i}\nu_1)]
/(2\pi/\beta)\\) (reduced indices \\(n = 1, 3\\)), and a negative
\\(Z\\) is reported as zero. Here is that
solution:

![Im G and Im Σ against ν](dmft_ipt_solution.png)

It looks entirely reasonable. It is also wrong. Leave the same loop running
with no stopping rule at all and the residual does this:

![The residual against the iteration](dmft_ipt_convergence.png)

The change of \\(\Sigma\\) dips below \\(10^{-5}\\) at iteration 65 — that is
where the criterion fires — then climbs back to order one around iteration
110 and only afterwards settles, reaching \\(10^{-16}\\) by iteration 1200.
The fixed point it settles on has \\(Z = 0\\): at \\(U = 5\\) and
\\(T = 0.05\\) this model is a Mott insulator, not a metal.

Nothing went wrong numerically. The trajectory passes close to the *unstable*
fixed point that separates the two solutions, and near it the step size is
tiny while the distance still to travel is not. A criterion that measures how
much \\(\Sigma\\) moved cannot tell a plateau from an answer. The escape is
also the one place in this calculation where rounding is visible: a
\\(10^{-16}\\) difference grows by some 40% per iteration while the
trajectory is being pushed away from the unstable point, so the iteration at
which it leaves (around 109 here) can move by a few steps between machines or
compilers. The shape of the curve — the rebound above 0.1, the final residual
below \\(10^{-15}\\) — does not.

The moral is not specific to IR: any self-consistency loop near a phase
boundary can stop on a plateau. The cure used below is the blunt one, and it
works because the basis makes iterations cheap.

## Scanning \\(U\\): the Mott transition and its hysteresis

`dmft_ipt_scan` sweeps \\(U\\) from 0 to 6.5 in 66 steps, three times:

- starting each \\(U\\) from the non-interacting \\(G^0\\);
- sweeping upwards, starting each \\(U\\) from the previous solution (the
  *metal* branch);
- sweeping downwards from \\(U = 6.5\\) (the *insulator* branch).

Every solve runs a fixed 5000 iterations with no stopping rule, for the
reason above. Nothing on the grid takes anywhere near that long to settle —
the slowest escape observed was between iterations 2000 and 4000, at the edge
of the coexistence window. With the criterion removed, the three branches no
longer depend on where exactly a threshold happened to be crossed.

![Z against U for the three sweeps](dmft_ipt_scan_renormalisation.png)

\\(Z\\) falls smoothly from 1 and then drops to zero, but not at the same
\\(U\\) going up as coming down. Below \\(U_{c1} = 3.2\\) only the metal
exists; above \\(U_{c2}\\), which lies between 3.4 and 3.5, only the
insulator does; in between both are stable and which one you get depends on
where you started. That shaded window is the first-order Mott transition, and
it is not an artifact of stopping early: solutions inside it sit at a
residual of \\(5\times10^{-17}\\) and stay there for thousands of further
iterations.

Starting from \\(G^0\\) lands on the metal branch wherever the metal exists —
the `from_g0` and `metal` columns agree to \\(10^{-15}\\) across the whole
grid — which is why a naive scan sees only \\(U_{c2}\\) and misses the
hysteresis entirely.

The self-energies on either side make the distinction concrete:

![Im Σ against ν for five values of U](dmft_ipt_scan_self_energy.png)

These are the solves started from \\(G^0\\), so they are on the metal branch
wherever it exists. At \\(U = 3.0, 3.2, 3.4\\) — the last two inside the
coexistence window — \\(\mathrm{Im}\,\Sigma\\) heads to zero with
\\(\nu\\), a Fermi liquid. At \\(U = 3.5\\) and \\(U = 4.0\\) it
diverges as \\(\nu \to 0\\), which is the pole at the Fermi level that opens
the Mott gap.

One curiosity worth naming, since it is visible in the raw output: the
insulating fixed point comes as a particle-hole-conjugate pair, and which of
the two a run lands on is decided by rounding. \\(\mathrm{Re}\,\Sigma\\)
flips sign between the two; \\(\mathrm{Im}\,\Sigma\\), \\(G\\) and \\(Z\\) do
not, so nothing physical depends on it.

## Beyond the imaginary axis

Everything above lives on the imaginary axis. The contrast between the metal
and the Mott insulator is most visible in the spectral function
\\(A(\omega)\\) — a quasiparticle peak at \\(\omega = 0\\) against a gap
between two Hubbard bands — and getting there from \\(G(\mathrm{i}\nu)\\) is
analytic continuation. The IR coefficients \\(g_l\\) computed here are the
natural input: see [analytic continuation](analytic_continuation.md) for
regularised inversion of \\(g_l = -s_l\rho_l\\),
[sparse modeling](spm.md) for the \\(\ell_1\\)-regularised version, and
[MiniPole](minipole.md) for a pole representation of
\\(G(\mathrm{i}\nu)\\) from which \\(A(\omega)\\) can be read off.

## Running it

From `docs/tutorial-code`:

```console
$ cargo run --profile ci --bin dmft_ipt
$ cargo run --profile ci --bin dmft_ipt_scan
$ uv run --project ../plotting python ../plotting/dmft_ipt_plot.py
```

`dmft_ipt` takes about a second. `dmft_ipt_scan` is 198 self-consistent
solves of 5000 iterations each, which is closer to half a minute. The
binaries write CSV tables under `docs/tutorial-code/data/`; the plotting
script only reads them.

The repository checks these numbers against the Python notebook. Because of
the rounding-sensitive escape described above, the 65-iteration solve is
compared to \\(5\times10^{-6}\\) rather than to machine precision, and the long
run by its shape rather than iteration by iteration.

## Key API pieces

From `sparse-ir`:

| What you want | What to call |
| --- | --- |
| the fermionic IR basis at \\(\beta\\), \\(\omega_\mathrm{max}\\), \\(\varepsilon\\) | `LogisticKernel::new`, `FiniteTempBasis::<_, Fermionic>::new` |
| sampling in \\(\tau\\) and \\(\mathrm{i}\nu\\) | `TauSampling::new`, `MatsubaraSampling::new` (wrapped by `IrMesh`) |
| \\(s_l\\) and \\(v_l(\omega)\\) for \\(g_l = -s_l\rho_l\\) | `basis.s()`, `basis.v()` (`get_knots`, `get_polyorder`) |
| the reduced index of a sampling frequency | `MatsubaraFreq::n` |

From the tutorial crate (`docs/tutorial-code/src`):

| What you want | Tutorial helper |
| --- | --- |
| \\(\tau \to \mathrm{i}\nu\\) and back, once per iteration | `IrMesh::tau_to_wn`, `IrMesh::wn_to_tau` |
| \\(\rho_l\\) of a semicircle, by quadrature | `shifted_semicircle_overlaps`, then `IrMesh::l_to_wn` |
| \\(\Sigma(\tau)\\) for plotting | `IrMesh::wn_to_l`, then `IrMesh::l_to_tau` |
| the lowest Matsubara frequency | the position of \\(n = 1\\) in `IrMesh::wn` |
