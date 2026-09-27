# Orbital magnetic susceptibility

*Ported from the Python notebook `orbital_magnetic_susceptibility_py.ipynb` of
sparse-ir-tutorial, whose authors are Soshun Ozaki and Takashi Koretsune. The
program that produced every number and figure on this page is
`docs/tutorial-code/src/bin/orbital_magnetic_susceptibility.rs`.*

The previous page used the basis to turn a Matsubara sum into one evaluation.
This one does the same thing for a harder summand, and adds the other piece
every multi-orbital calculation needs: a Hamiltonian that is a matrix at each
momentum.

## Theory

Orbital magnetism is what a vector potential coupled to the electron momentum
induces. For a tight-binding model the susceptibility has a closed form in
terms of the Green's function and the derivatives of the Hamiltonian,

\\[
\chi = T \sum_\nu \chi(\mathrm{i}\nu), \qquad
\chi(\mathrm{i}\nu) = \sum_{\boldsymbol{k}} \mathrm{Tr}\left[
  \gamma_x G \gamma_y G \gamma_x G \gamma_y G
  + \tfrac{1}{2}(\gamma_x G \gamma_y G + \gamma_y G \gamma_x G)\gamma_{xy} G
\right],
\\]

with \\(G = G(\mathrm{i}\nu, \boldsymbol{k})\\),
\\(\gamma_i = \partial H_{\boldsymbol{k}} / \partial k_i\\) and
\\(\gamma_{xy} = \partial^2 H_{\boldsymbol{k}} / \partial k_x \partial k_y\\).
The formula is due to Gómez-Santos and Stauber, and to Raoux, Piéchon, Fuchs
and Montambaux; its relation to Fukuyama's continuum formula was settled by
Ogata and Fukuyama and by Matsuura and Ogata.

It is written entirely in Matsubara sums of products of Green's functions,
which is what makes it a good example. Here the summand is a product of three
or four of them, not two:

![The summand against frequency](orbital_magnetic_susceptibility_summand.png)

The square lattice falls off as \\(\nu^{-4}\\) — a dispersion that separates
has no \\(\gamma_{xy}\\), so the three-Green's-function term is absent
entirely — and graphene, whose velocity matrices are purely off-diagonal in
the site basis, as \\(\nu^{-6}\\). Thirty sampling frequencies hold all of it,
and the sum is again

```rust,ignore
let u_at_zero = basis.evaluate_tau(&[0.0])?;

let coefficients = mesh.wn_to_l(&chi_iw, N_MU)?;
let chi = evaluate_rows(&u_at_zero, &coefficients, N_MU);
```

with one column per chemical potential; there are 91 of them, fitted in one
call.

## Square lattice

One band makes every matrix in the formula a number and the trace a product,
so the whole summand is two terms:

```rust,ignore
let g = Complex64::new(-(ek - mu), nu).inv();
let g2 = g * g;
chi[i * n_mu + m] += gx * gx * gy * gy * g2 * g2 + gx * gy * gxy * g2 * g;
```

\\(\gamma_{xy}\\) is identically zero here, and the second term with it; the
example keeps it so that the code is the formula rather than a special case
of it.

![χ against μ for the square lattice](orbital_magnetic_susceptibility_square.png)

At \\(T = 0\\) the same quantity is known in closed form,

\\[
\chi = -\frac{2}{3\pi^2}\left[E(m) - \frac{K(m)}{2}\right],
\qquad m = 1 - \frac{\mu^2}{16},
\\]

zero outside the band. `sparse_ir_tutorial::elliptic` computes \\(K\\) and
\\(E\\) from the arithmetic-geometric mean, which is a dozen lines and needs
no dependency. The two curves lie on top of each other to better than
\\(10^{-3}\\) for \\(1 \le |\mu| \le 3\\), which the example asserts, and part
company only where they must: at the van Hove filling \\(\mu = 0\\), where the
closed form diverges and \\(T = 0.1\\) does not, and at the band edge, where
the Fermi function rounds the step off. The divergence is also why
\\(\mu = 0\\) is missing from the reference values of the closed form — there
is no finite number there to compare against.

## Graphene

Two sites per cell make \\(H_{\boldsymbol{k}}\\) a 2×2 matrix, which has to be
diagonalised before the Green's function can be written down. In the
eigenbasis \\(G\\) is diagonal, so the trace of four matrices becomes a
product of 2×2 matrices whose columns are scaled by \\(G\\):

```rust,ignore
let eigen = hamiltonian.eigen();
let [gx, gy, gxy] = velocities.map(|m| Hermitian2::rotate(&eigen, &m));
...
let (xg, yg, xyg) = (times_green(&gx, &g), times_green(&gy, &g), times_green(&gxy, &g));
let xy = product(&xg, &yg);
let yx = product(&yg, &xg);
chi[i * n_mu + m] += trace(&xy, &xy) + 0.5 * (trace(&xy, &xyg) + trace(&yx, &xyg));
```

`Hermitian2` in `sparse_ir_tutorial::linalg` solves the 2×2 eigenproblem in
closed form rather than calling a general routine, because a general routine
is free to return the eigenvectors with any phase and the tutorial's numbers
have to be reproducible. The susceptibility does not care about the phase —
it is a trace of rotated matrices — and the agreement with numpy's `eigh` to
\\(2 \times 10^{-14}\\) in the verification test is what says so.

![χ against μ for graphene](orbital_magnetic_susceptibility_graphene.png)

The result is the opposite of the square lattice: a sharp *diamagnetic* peak
at the Dirac point, the delta function of the massless spectrum rounded by
temperature, paramagnetic shoulders on either side, and nothing at all outside
the band. The example asserts each of those three statements.

## Key API pieces

| What you want | What to call |
| --- | --- |
| a Matsubara sum | fit, then evaluate at `τ = 0` |
| many chemical potentials at once | one column each; `wn_to_l` takes them together |
| a 2×2 Hermitian eigenproblem | `Hermitian2::eigen`, `Hermitian2::rotate` |
| \\(K(m)\\), \\(E(m)\\) | `elliptic::ellipk`, `elliptic::ellipe` |
