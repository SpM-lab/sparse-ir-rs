# Sparse modeling

*Ported from the Python notebook `spm_py.ipynb` of sparse-ir-tutorial. The
program that produced every number and figure on this page is
`docs/tutorial-code/src/bin/spm.rs`.*

Everything so far went the easy way: from a spectral function to `G`. This
page goes back — from a noisy \\(G(\tau)\\) to the \\(\rho(\omega)\\) that
produced it. That is analytic continuation, and it is ill posed: the kernel
smooths, so the inverse sharpens, and it sharpens the noise along with the
signal.

The IR basis makes the difficulty explicit rather than making it go away. In
the basis,

\\[
    G_l = -s_l \rho_l,
\\]

so recovering \\(\rho_l\\) means dividing by \\(s_l\\). The singular values
fall off exponentially, so past the point where \\(s_l\\) drops below the noise
level, the data say nothing at all about \\(\rho_l\\). *Sparse modeling* is the
choice to set those coefficients to zero rather than to let them be whatever
the noise dictates.

## The problem

Minimise, over the IR coefficients \\(x_l = \rho_l\\),

\\[
    \tfrac12 \lVert y - A x \rVert^2 + \lambda \lVert x \rVert_1,
    \qquad y_i = -G(\tau_i), \quad A_{il} = u_l(\tau_i)\, s_l .
\\]

The first term says the answer must explain the data. The L1 penalty is what
makes the solution *sparse*: unlike a least-squares penalty, it drives
coefficients to exactly zero instead of merely making them small. \\(\lambda\\)
decides how many survive.

## The data

The notebook downloads a sample `Gtau.in` from the SpM repository. A tutorial
that builds in CI must not fetch anything at run time, so this port ships its
input instead: `docs/tutorial-code/input/spm/gtau.csv`, written once by
`scripts/make_spm_input.py`. It is \\(G(\tau)\\) of the three-Gaussian spectral
function from [the transformation page](transformation.md), at
\\(\beta = 100\\) and \\(\omega_\mathrm{max} = 4\\), on 401 uniform times, plus
independent Gaussian noise of size \\(10^{-3}\\) from a fixed seed. Committing
it means the Rust example and its Python reference start from bit-identical
numbers — and, unlike the downloaded file, it comes with the exact answer.

![The data](spm_gtau.png)

The noise is invisible here — \\(10^{-3}\\) against a curve of order 1 — and it
is what decides how much of the spectrum can be recovered.

```rust,ignore
let input = read_table(&input_path("spm", "gtau"))?;
let taus = input.expect_column("tau").to_vec();
let g_tau = input.expect_column("g_tau").to_vec();
```

## Building `A`

\\(A\\) is the matrix of basis functions at the sampled times, scaled by the
singular values. `evaluate_tau` returns \\(u_l(\tau_i)\\) with the points along
the first axis:

```rust
use sparse_ir::{Basis, Fermionic, FiniteTempBasis, LogisticKernel, Matrix};

let beta = 100.0;
let wmax = 4.0;
let kernel = LogisticKernel::new(beta * wmax)?;
let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(1e-10), None)?;

let taus: Vec<f64> = (0..401).map(|i| beta * i as f64 / 400.0).collect();
let u: Matrix<f64> = basis.evaluate_tau(&taus)?;

let mut a = vec![0.0; taus.len() * basis.size()];
for i in 0..taus.len() {
    for l in 0..basis.size() {
        a[i * basis.size() + l] = u.get(&[i, l]).unwrap() * basis.s()[l];
    }
}

assert_eq!(basis.size(), 43);
# Ok::<(), sparse_ir::Error>(())
```

43 unknowns against 401 equations — and still ill posed, because the last
columns of \\(A\\) are multiplied by singular values of order \\(10^{-10}\\).

## Solving it

`sparse_ir_tutorial::fista` is the standard proximal-gradient method: a
gradient step on the smooth part, then the proximal operator of the penalty,
with Nesterov momentum in between. For an L1 penalty the proximal operator is
soft thresholding — shrink every coefficient towards zero by a fixed amount and
clip the ones that would cross.

```rust,ignore
use sparse_ir_tutorial::{fista, soft_threshold};

let report = fista(
    &mut x,
    lipschitz,
    |x, grad| { /* grad = Aᵀ(Ax − y) */ },
    |x, step| soft_threshold(x, step * lambda),
    20_000,
    0.0,
);
```

Two details matter for reproducibility. The step length is \\(1/L\\) with
\\(L\\) the largest eigenvalue of \\(A^\mathsf{T} A\\), computed by a
power iteration with a *fixed* number of steps from a fixed start; and the
iteration runs for a fixed budget rather than stopping on a tolerance. FISTA's
momentum makes the step-to-step change oscillate rather than fall
monotonically, so a tolerance is either never reached or reached at an
iteration a rounding difference can move. Fixing both keeps this program and
its Python reference on the same trajectory, which is why they agree to
\\(10^{-14}\\) rather than to the \\(10^{-6}\\) an iterative method would
otherwise justify.

## What comes out

![The recovered spectrum](spm_spectrum.png)

All three peaks come back, the narrow central one included, from data whose
noise is a thousand times smaller than the signal but ten million times larger
than the smallest singular value that matters.

![The surviving coefficients](spm_coefficients.png)

This is the mechanism, in one picture. Only 11 of the 43 coefficients are
nonzero; the rest are *exactly* zero, not small. The cut-off sits where
\\(s_l\\) falls to about \\(10^{-3}\\) — the noise level. Below that the data
carry no information about \\(\rho_l\\), and the L1 penalty declines to invent
any.

## Choosing λ

![Residual and error against λ](spm_lambda_scan.png)

The residual sits flat at the noise level, \\(\lVert \delta G \rVert \approx
10^{-3}\sqrt{401} \approx 0.021\\), until \\(\lambda \approx 10^{-3}\\), and
then rises: past that point the penalty is discarding signal, not noise. The
error against the exact spectrum bottoms out earlier, at
\\(\lambda = 3\times10^{-5}\\), which is the value this page reports.

The two do not coincide, and that is the honest situation: the error curve
needs the answer you are trying to find. In practice you have only the residual
curve and the noise level, which together bracket \\(\lambda\\) to about a
decade — and within that decade the spectrum barely changes.

![Nonzero coefficients against λ](spm_sparsity.png)

## What this port leaves out

The notebook solves a *constrained* version of the same problem: it adds
\\(\rho(\omega) \ge 0\\) and the sum rule \\(\int\mathrm{d}\omega\,\rho = 1\\)
as hard constraints. Neither has a cheap proximal operator — \\(\rho \ge 0\\)
is a constraint on \\(V x\\), not on \\(x\\) — so the notebook reaches for ADMM
and the `admmsolver` package. This port keeps the solver to twenty lines of
FISTA and drops both constraints.

The output says what that costs. The recovered spectrum dips to
\\(-7\times10^{-4}\\) — visible in the figure as the curve crossing zero
between the peaks — and the sum rule comes out at 1.0029 rather than exactly 1.
Both are small because the L1 fit is already close to the truth, not because
anything enforced them. If you need a spectrum that is non-negative by
construction, you need the constrained solver.

## Key API pieces

| What you want | What to call |
| --- | --- |
| \\(u_l(\tau_i)\\) for arbitrary times | `Basis::evaluate_tau` |
| \\(v_l(\omega_j)\\) for arbitrary frequencies | `Basis::evaluate_omega` |
| the singular values | `FiniteTempBasis::s` |
| the solver | `sparse_ir_tutorial::fista`, `soft_threshold` (this tutorial, not the library) |

`fista` and `soft_threshold` live in the tutorial's own crate. `sparse-ir`
gives you the basis and the transforms; what you do with them is yours.
