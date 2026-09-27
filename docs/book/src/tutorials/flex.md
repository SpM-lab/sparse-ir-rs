# Fluctuation exchange

*Ported from the Python notebook `FLEX_py.ipynb` of sparse-ir-tutorial, whose
author is Niklas Witt. The programs that produced every number and figure on
this page are `docs/tutorial-code/src/bin/flex.rs` and
`docs/tutorial-code/src/bin/flex_scan.rs`.*

The previous page fixed its vertices by solving two scalar equations and never
iterated. The fluctuation-exchange approximation goes the other way: the
interaction stays the bare \\(U\\), and everything is decided by a Dyson loop
that is run until the self-energy stops moving. What makes it belong here is
that each pass of that loop is a handful of products in \\((\tau, r)\\) with
basis transforms between them — the frequency dependence never appears as a
truncated sum.

## One loop, four transforms

FLEX sums the particle-hole bubble and ladder series, which in a single band
collapses to a self-energy built from the same irreducible \\(\chi^0\\) as on
the TPSC page, dressed twice:

\\[
\chi_\mathrm{sp} = \frac{\chi^0}{1 - U\chi^0},
\qquad
\chi_\mathrm{ch} = \frac{\chi^0}{1 + U\chi^0},
\qquad
V = U^2\left(\tfrac32 \chi_\mathrm{sp} + \tfrac12 \chi_\mathrm{ch}
            - \chi^0\right),
\\]

\\[
\Sigma(\tau, r) = V(\tau, r)\, G(\tau, r),
\qquad
G(\mathrm{i}\nu, k) = \bigl[\mathrm{i}\nu - (\varepsilon_k - \mu)
                            - \Sigma(\mathrm{i}\nu, k)\bigr]^{-1}.
\\]

Read as code, one step is: dress \\(\chi^0\\) in \\((\mathrm{i}\nu^B, q)\\),
carry \\(V\\) to \\((\tau, r)\\), multiply by \\(G(\tau, r)\\), carry the
product back to \\((\mathrm{i}\nu, k)\\).

```rust,ignore
let v: Vec<Complex64> = (0..ckio.len())
    .map(|i| u * u * (1.5 * chi_spin[i] + 0.5 * chi_charge[i] - ckio[i]))
    .collect();
let interaction = mesh_b.wn_to_tau(&grid.k_to_r(&v), nk)?;

let product: Vec<Complex64> =
    interaction.iter().zip(&grit).map(|(v, g)| v * g).collect();
let sigma = mesh_f.tau_to_wn(&grid.r_to_k(&product), nk)?;
```

The bosonic mesh fits the interaction and the fermionic one fits the
self-energy, out of the same SVE and therefore on the same \\(\tau\\) grid —
the point `tpsc.rs` already depended on for `reverse_tau`.

The constant Hartree term \\(\sim U n/2\\) is left out of \\(V\\). It is
frequency independent, so the basis is the wrong tool for it, and in a single
band it is a shift of the chemical potential, which is refitted at every step
anyway. That omission comes back in the gap equation below.

## Keeping the denominator positive

Nothing forces \\(1 - U\chi^0\\) to stay positive along the way to the fixed
point, and \\(\chi^0\\) evaluated on the *non-interacting* Green's function is
the largest it will ever be. At \\(U = 4\\) the starting point does violate
it, so the interaction is walked up instead of being switched on at once:

```rust,ignore
while target * self.max_chi() >= 1.0 {
    self.u = target / (self.max_chi() * target + 0.01);
    self.step()?;
    self.u = target;
}
```

Each pass runs one FLEX step at the largest interaction the current
\\(\chi^0\\) can carry, which shrinks \\(\chi^0\\); the target \\(U\\) is then
retried. The loop is capped, because there is no guarantee it ever succeeds:
a genuinely ordered system has no FLEX solution to walk towards. At
\\(T = 0.1\\) six passes are enough, and the number is written out
(`renormalisation_steps`) so a change in it fails the verification test rather
than quietly shifting the answer.

## One solve at U = 4

`flex` takes a \\(24 \times 24\\) lattice at \\(\beta = 10\\), filling
\\(n = 0.85\\), \\(U = 4\\), \\(\varepsilon = 10^{-10}\\) and a linear mixing
of \\(0.2\\), and runs 30 steps. The basis has 30 functions for each
statistics: 30 \\(\tau\\) points, 30 fermionic and 31 bosonic frequencies. The
relative movement of \\(\Sigma\\) in the last step is \\(1.3\times10^{-5}\\),
and \\(\mu = -0.6713\\).

![Im Σ, χ_sp and Δ over the Brillouin zone](flex_zone.png)

\\(|\mathrm{Im}\,\Sigma|\\) is largest at the antinodes \\((\pi, 0)\\) and
smallest along the diagonal — the momentum-selective scattering that opens a
pseudogap — and \\(\chi_\mathrm{sp}\\) is piled up at \\(M = (\pi, \pi)\\),
enhanced there by a factor of \\(17.5\\) over \\(\chi^0\\) against \\(2.3\\)
at \\(\Gamma\\).

![susceptibilities along Γ→X→M→Γ, and Σ and Δ at the antinode](flex_path.png)

Against frequency, \\(\mathrm{Im}\,\Sigma\\) is odd, largest in magnitude at
the first Matsubara frequency and decaying from there; the run gives
\\(\mathrm{Im}\,\Sigma(\mathrm{i}\pi T, (\pi,0)) = -0.588\\). The verification
test checks the sign at every frequency, which is the statement that the
solution is causal.

## The linearised gap equation

With the fluctuations in hand, the question is whether they pair. The
linearised Eliashberg equation

\\[
\lambda\, \Delta(\mathrm{i}\nu, k) = -\frac{T}{N_k}
\sum_{\mathrm{i}\nu', k'} V^S(\mathrm{i}\nu - \mathrm{i}\nu', k - k')\,
F(\mathrm{i}\nu', k'),
\qquad
F = |G|^2\,\Delta,
\\]

is an eigenvalue problem whose leading \\(\lambda\\) reaches 1 at
\\(T_\mathrm{c}\\). The singlet vertex is *not* the one in the self-energy:
the charge fluctuations enter with the opposite sign,
\\(V^S = U^2\left(\tfrac32\chi_\mathrm{sp} - \tfrac12\chi_\mathrm{ch}\right)\\).
Applying it is the same four transforms as one FLEX step, so the power method
costs about what a few self-consistency steps cost, seeded with a
\\(d_{x^2-y^2}\\) gap \\(\cos k_x - \cos k_y\\).

Here the dropped Hartree term matters. It is exactly zero in the
\\(d\\)-wave channel, because such a gap sums to zero over the zone, but not
outside it, and without it the operator has a spurious mode whose eigenvalue
is \\(-1.28\\) — larger in magnitude than \\(\lambda_d\\). The seed has almost
no overlap with it, so the iterate sits on the \\(d\\)-wave answer for tens of
steps and then leaves it. A fixed step count would land right at that
crossover; the loop therefore stops when \\(\lambda\\) moves by less than
\\(10^{-4}\\), which happens after 8 steps with three orders of magnitude to
spare, and the step count is written out so that a divergence between
implementations cannot pass unnoticed.

The result is \\(\lambda_d = 0.4568\\) at \\(T = 0.1\\): the right symmetry,
well short of the transition. The gap in the third panel above changes sign
under \\(k_x \leftrightarrow k_y\\) and vanishes on the zone diagonal, which
is what the verification test asserts — the eigenvector's overall scale means
nothing, only its shape does.

## Cooling towards T_c

`flex_scan` repeats the calculation on a \\(64 \times 64\\) lattice at seven
temperatures from \\(T = 0.08\\) down to \\(0.025\\), with
\\(\Lambda = \beta\omega_\mathrm{max}\\) held at \\(10^3\\) and
\\(\varepsilon = 10^{-8}\\). Holding \\(\Lambda\\) fixed means one SVE serves
every temperature —

```rust,ignore
let (kernel, sve) = sve_for(beta_init, LAMBDA / beta_init, EPS)?;
let lattice = Lattice::from_sve(kernel, sve.clone(), NK_LIN, NK_LIN, T_HOP, beta, EPS)?;
```

— and that every temperature has a basis of the same 43 functions, which in
turn is what makes it legal to carry the converged \\(\Sigma\\) of one
temperature into the next as a starting point. With that head start, the
renormalisation loop runs only at the first temperature (nine passes) and
never again.

![1/χ_sp and λ_d against temperature](flex_scan_lambda.png)

\\(1/\chi_\mathrm{sp}^{\max}\\) falls almost linearly — the Curie–Weiss form —
from \\(0.1996\\) to \\(0.0248\\), extrapolating to zero near \\(T
\approx 0.019\\), while \\(\lambda_d\\) climbs from \\(0.533\\) to
\\(0.955\\) and would reach 1 at about \\(T \approx 0.021\\). The two
temperatures are close because in this approximation it is the spin
fluctuations that do the pairing: the gap equation is driven by the same
\\(\chi_\mathrm{sp}\\) that is diverging.

![χ_sp along the path at T = 0.03](flex_scan_chi_spin.png)

The finer grid also resolves what the \\(24\times24\\) run could not: at
\\(n = 0.85\\) the peak is *not* at \\(M\\). Nesting is incommensurate, and
\\(\chi_\mathrm{sp}\\) peaks at \\((\pi, 7\pi/8)\\) with the value \\(24.8\\)
against \\(5.3\\) at \\(M\\) itself — a factor of nearly five that a
commensurate grid would have missed entirely.

## Running it

`flex` is one solve and takes under half a second; `flex_scan` is seven solves
on a grid seven times larger and takes a few seconds, so it is registered as a
parameter scan:

```console
$ SPARSEIR_TUTORIAL_RUN=1 SPARSEIR_TUTORIAL_SCANS=1 cargo test --release \
      --test tutorial_binaries --test verification
```

or `scripts/check.sh --run --scans --release`.

## Key API pieces

| What you want | What to call |
| --- | --- |
| one step of a Dyson loop | `IrMesh::wn_to_tau`, `MomentumGrid::k_to_r`, multiply, `MomentumGrid::r_to_k`, `IrMesh::tau_to_wn` |
| \\(\chi^0\\) as a real-space product | `IrMesh::reverse_tau`, then the bosonic `IrMesh::tau_to_wn` |
| the filling from \\(G(\tau = 0^-)\\) | `IrMesh::wn_to_l`, then `evaluate_rows` on \\(U^F_\ell(0)\\) |
| the chemical potential at fixed \\(n\\) | `brent` from `tutorial::roots` |
| one SVE reused across temperatures | `sve_for`, then `Lattice::from_sve` per \\(\beta\\) |
| a whole basis at fixed \\(\Lambda\\) | keep \\(\beta\omega_\mathrm{max}\\) constant and vary \\(\beta\\) |
