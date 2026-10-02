# Analytic continuation

*Ported from the Python notebook
[`analytic_continuation_py.ipynb`](https://spm-lab.github.io/sparse-ir-tutorial-v2/src/analytic_continuation_py.html)
of [sparse-ir-tutorial-v2](https://spm-lab.github.io/sparse-ir-tutorial-v2/).
The program that produced every number and figure on this page is
`docs/tutorial-code/src/bin/analytic_continuation.rs`; the code below is
included from it.*

Going from a spectral function to a Green's function is a smoothing integral
and always works. This page looks at the inverse, from \\(G\\) back to
\\(\rho(\omega)\\), and shows why it is ill posed: what fails, what two simple
regularisers buy, and why the IR coefficients of \\(\rho\\) are the wrong
unknowns. [Sparse modeling](spm.md) then solves one such problem in full, and
[MiniPole](minipole.md) takes a different route by fitting poles.

## The problem

\\[
    G(\tau) = -\int \mathrm{d}\omega\, K(\tau, \omega)\, \rho(\omega)
\\]

In the basis this reads \\(G_l = -s_l \rho_l\\), one number at a time, and it
inverts in one line:

\\[
    \rho(\omega) = -\sum_l \frac{G_l}{s_l} v_l(\omega).
\\]

Every \\(s_l\\) is strictly positive, so the inverse exists. It is also
useless, because the \\(s_l\\) fall off exponentially:

```rust
{{#include ../../../tutorial-code/src/bin/analytic_continuation.rs:imports}}
{{#include ../../../tutorial-code/src/bin/analytic_continuation.rs:parameters}}
{{#include ../../../tutorial-code/src/bin/analytic_continuation.rs:basis}}
# Ok::<(), Box<dyn std::error::Error>>(())
```

![The singular values](analytic_continuation_singular_values.png)

At \\(\beta = 40\\), \\(\omega_\mathrm{max} = 2\\) and
\\(\varepsilon = 2 \times 10^{-8}\\) the basis has 24 functions and
\\(s_{23}/s_0 \approx 2.5 \times 10^{-8}\\). Dividing by that last singular
value multiplies whatever error rides on \\(G_{23}\\) by forty million. There
is no algorithm that avoids this: the information about the fine structure of
\\(\rho\\) is not in \\(G\\) to be recovered. All a method can do is choose
what to put in its place.

## Two models

Both are sums of normalised semicircles, so their exact \\(\rho_l\\) can be
computed to machine precision and used as ground truth. One fills the band;
the other is the same band split by a gap.

![The two models](analytic_continuation_models.png)

`shifted_semicircle_overlaps` (a tutorial helper) computes \\(\rho_l\\) of one
semicircle by quadrature, and \\(G_l = -s_l \rho_l\\):

```rust,ignore
{{#include ../../../tutorial-code/src/bin/analytic_continuation.rs:models}}
{{#include ../../../tutorial-code/src/bin/analytic_continuation.rs:coefficients}}
```

Then noise is added to \\(G_l\\), at \\(0.3\,(s_{L-1}/s_0)\\) of
\\(\lVert G \rVert\\) — deliberately of the same size as the smallest
coefficient the basis still resolves.

![The noisy coefficients](analytic_continuation_noisy_coefficients.png)

The noise is far below every coefficient that matters and level with the ones
at the end. The standard-normal draws are read from
`docs/tutorial-code/input/analytic_continuation/noise.csv`.

## Truncated SVD

The first regulariser is the blunt one: keep \\(\rho_l = -G_l/s_l\\) for
\\(l < L'\\) and throw the rest away.

![Truncated SVD, semi-elliptic](analytic_continuation_tsvd_semielliptic.png)

\\(L' = 12\\) gets the shape right. \\(L' = 24\\) — using everything the basis
has — turns the answer into ringing of amplitude 0.2 around a function whose
peak is 0.32. Using *more* of the data made the answer worse, which is the
whole difficulty in one picture.

The insulating model says the same thing with a sharper edge:

![Truncated SVD, insulating](analytic_continuation_tsvd_insulating.png)

## Ridge regression

The second regulariser is smooth: minimise
\\(\lVert G - (-s\rho) \rVert^2 + \alpha^2 \lVert \rho \rVert^2\\). Because
the problem is diagonal in the basis, the solution is diagonal too,

\\[
    \rho_l = -\frac{s_l}{s_l^2 + \alpha^2}\, G_l,
\\]

which passes the large singular values through unchanged and rolls the small
ones off instead of dividing by them.

```rust,ignore
{{#include ../../../tutorial-code/src/bin/analytic_continuation.rs:ridge}}
```

![Ridge regression](analytic_continuation_ridge.png)

With \\(\alpha = 100 \times \text{noise}\\) both models come back recognisably.
Note what is left over: the reconstruction goes negative inside the gap. A
density of states cannot do that, and no amount of tuning \\(\alpha\\) will
stop it, because nothing in the method knows that \\(\rho \geq 0\\).

## Why ρ<sub>l</sub> is the wrong unknown

It is tempting to conclude that the fix is a better penalty on \\(\rho_l\\).
It is not, and the reason is worth stating plainly: \\(G_l\\) is compact,
\\(\rho_l\\) is not. The basis is small because \\(s_l\\) makes it small — and
\\(s_l\\) belongs to \\(G\\), not to \\(\rho\\).

Take four delta peaks, \\(\rho(\omega) = \sum_i \delta(\omega - \omega_i)\\),
so that \\(\rho_l = \sum_i v_l(\omega_i)\\):

![The coefficients of a discrete spectrum](analytic_continuation_discrete.png)

\\(G_l\\) falls off the way it always does. \\(\rho_l\\) does not fall off at
all — it is still of order one at \\(l = 23\\), and would be at \\(l = 200\\).
Truncating it at \\(L'\\) is not an approximation of anything; a penalty on
\\(\sum_l |\rho_l|^2\\) is a penalty on coefficients that were never going to
become small.

A spectrum made of a few poles is, on the other hand, exactly the case that
pole-fitting methods handle well. [MiniPole](minipole.md) uses ESPRIT to fit a
small number of poles and residues directly to Matsubara data, instead of
expanding \\(\rho\\) in any basis.

## A real-axis basis instead

So expand \\(\rho\\) on the real axis rather than in the IR basis:

\\[
    \rho(\omega) = \sum_{m} a_m\, f(\omega - \omega_m),
    \qquad
    f(\omega) = \frac{1}{\pi}\frac{\eta}{\omega^2 + \eta^2}.
\\]

![The real-axis basis function](analytic_continuation_lorentz.png)

\\(f\\) is a probability density, so \\(a_m \geq 0\\) gives \\(\rho \geq 0\\)
and \\(\sum_m a_m = 1\\) gives the sum rule — both of them constraints a
solver can impose exactly, neither of them expressible in the IR coefficients.
The width \\(\eta\\) is what is left to choose; \\(\eta = 0.1\,\pi/\beta\\) is
narrow enough to resolve structure on the scale of the temperature and wide
enough not to be a set of delta functions.

The data then enter through

\\[
    G_l = \sum_m K_{lm} a_m,
    \qquad
    K_{lm} = -s_l \int \mathrm{d}\omega\, v_l(\omega) f(\omega - \omega_m),
\\]

which is an ordinary constrained least-squares problem in \\(a\\).

![The kernel of the real-axis basis](analytic_continuation_lorentz_kernel.png)

Computing \\(K_{lm}\\) is the one place this example does real numerical work.
\\(\eta\\) is about a hundredth of the spacing between the basis' knots, so
quadrature on the knots steps straight over the peak of \\(f\\). The
substitution \\(\omega = \omega_m + \eta \tan t\\) turns
\\(f(\omega - \omega_m)\,\mathrm{d}\omega\\) into \\(\mathrm{d}t/\pi\\) and
spreads the peak across the whole interval, leaving a polynomial to integrate:

```rust,ignore
{{#include ../../../tutorial-code/src/bin/analytic_continuation.rs:lorentz_overlaps}}
```

## What this page stops short of

This page does not solve the constrained least-squares problem in \\(a\\).
[Sparse modeling](spm.md) solves a related one: an L1-regularised fit in the
IR coefficients \\(\rho_l\\), *without* \\(\rho \geq 0\\) or the sum rule.
Solving on the Lorentzian basis with both constraints imposed is the subject
of the maximum-entropy and sparse-modeling literature, and is past where a
tutorial ends.

## Running it

From `docs/tutorial-code`:

```console
$ cargo run --release --bin analytic_continuation
```

The program writes CSV tables to `docs/tutorial-code/data/analytic_continuation/`.
The figures are drawn from those tables; from the repository root, run
`uv run --project docs/plotting python docs/plotting/analytic_continuation_plot.py`.
The noise is committed rather than drawn at run time, so that this program and
its Python counterpart,
`docs/tutorial-code/scripts/reference_analytic_continuation.py`, use the same
numbers; the header of `noise.csv` says where the draws came from.

## Key API pieces

From `sparse-ir`:

| What you want | What to call |
| --- | --- |
| the singular values | `FiniteTempBasis::s` |
| \\(v_l\\) on a grid of ω | `Basis::evaluate_omega` |
| the knots the basis is piecewise-polynomial on | `basis.v().get_knots(None)` |
| one \\(v_l\\) as a polynomial | `basis.v()[l].evaluate(omega)` |
| the degree to integrate exactly | `basis.v().get_polyorder()` |

From the tutorial crate (`sparse_ir_tutorial`, not part of the library):

| What you want | What to call |
| --- | --- |
| \\(\rho_l\\) of a semicircle | `shifted_semicircle_overlaps` |
| composite Gauss–Legendre quadrature over segments | `integrate_segments` |
