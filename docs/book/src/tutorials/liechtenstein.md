# Exchange interactions

*Ported from the Python notebook `liechtenstein_py.ipynb` of
sparse-ir-tutorial, whose author is Takuya Nomoto. The program that produced
every number and figure on this page is
`docs/tutorial-code/src/bin/liechtenstein.rs`.*

Both previous applied pages used the basis to hold a function of imaginary
time. This one uses it for something narrower and, in its way, more striking:
to do a Matsubara sum in one evaluation.

## Theory

The Liechtenstein method reads the parameters of a classical spin model off an
itinerant one by requiring that the two have the same second derivative of the
total energy with respect to spin angles. For a single orbital,

\\[
H = -t \sum_{\langle i, j\rangle, s} c^\dagger_{is} c_{js}
  - \sum_i \sum_{ss'} c^\dagger_{is}
    (\boldsymbol{B}_i \cdot \boldsymbol{\sigma}_{ss'}) c_{is'},
\\]

expanded about the ferromagnetic state \\(\boldsymbol{B}_i = B\hat{z}\\), the
requirement gives

\\[
J_{ij} = -B^2 T \sum_\nu G_{ij,+}(\mathrm{i}\nu) G_{ji,-}(\mathrm{i}\nu),
\\]

and, for the quantity that sets the mean-field transition temperature
\\(T_c^\mathrm{mf} = 2J_0/3\\),

\\[
J_0 = \frac{B}{2}(n_{0,+} - n_{0,-})
    + B^2 T \sum_\nu G_{00,+}(\mathrm{i}\nu) G_{00,-}(\mathrm{i}\nu).
\\]

Both are Matsubara sums of a *product* of two Green's functions. That product
falls off as \\(1/\nu^2\\), so a truncated sum converges like \\(1/N_M\\) —
slowly, and the more slowly the colder the system.

## The sum as an evaluation

The way out is the elementary identity

\\[
T \sum_\nu F(\mathrm{i}\nu) = F(\tau = 0),
\\]

which turns the sum into one point of the imaginary-time function behind it.
A product of two Green's functions is as representable in the basis as a
single one, so the whole sum costs a fit and one evaluation:

```rust,ignore
// T Σ_ν F(iν) = F(τ = 0): the row of basis functions that reads the sum off.
let u_at_zero = basis.evaluate_tau(&[0.0])?;

let coefficients = mesh.wn_to_l(&sampled, N_MU)?;
let summed = evaluate_rows(&u_at_zero, &coefficients, N_MU);
```

`sampled` holds the product at the 38 sampling frequencies, one column per
chemical potential; `evaluate_rows` contracts the coefficients with
\\(u_l(0)\\). At \\(\beta = 50\\) and \\(\Lambda = 800\\) the basis has 37
functions, and it would have about 37 at \\(\beta = 5000\\) too — the cost of
the sum does not grow with \\(\beta\\), only logarithmically.

The example also sums the series the naive way, on grids of 200 to 3200
frequencies, so that the two can be put side by side:

![J₀ against μ](liechtenstein_j0.png)

The truncated sums are above the converged answer everywhere and creep down
towards it. How fast is the whole story:

![The error against the grid size](liechtenstein_convergence.png)

Halving the error costs a doubling of the grid. Reaching the basis answer that
way would take of order \\(10^{9}\\) frequencies; the basis reaches it with
38.

The physics, once the answer is trusted: \\(J_0 < 0\\) at half filling, where
the system is an antiferromagnet by super-exchange, and \\(J_0 > 0\\) in the
low-carrier regime, where double exchange makes it a ferromagnet.

## \\(J_{ij}\\), and a check

\\(J_{ij}\\) needs the Green's function in real space rather than its zone
average, so a Fourier transform joins the sum. \\(G_{ij}\\) carries
\\(e^{-\mathrm{i}k\cdot r}\\) and \\(G_{ji}\\) carries
\\(e^{+\mathrm{i}k\cdot r}\\); since \\(\epsilon_{\boldsymbol{k}}\\) is even,
the second is the first applied to the same array, which the example asserts
rather than assumes:

```rust,ignore
assert!(
    /* ε(k) = ε(−k) */,
    "the dispersion must be even in k for G_ji to be the transform of G_ij"
);
```

![J_ij against distance](liechtenstein_jij.png)

The sum rule \\(J_0 = \sum_j J_{0j}\\) then connects the two calculations,
which took different routes: one never left momentum space, the other summed
1296 real-space terms. They agree to \\(4 \times 10^{-8}\\), which is the
accuracy of the basis (\\(\varepsilon = 10^{-7}\\)) and not machine precision
— exactly as it should be, and the example asserts that bound.

## Key API pieces

| What you want | What to call |
| --- | --- |
| a Matsubara sum | fit, then evaluate at `τ = 0` |
| the row \\(u_l(0)\\) | `Basis::evaluate_tau(&[0.0])` |
| contract it with a block of coefficients | `evaluate_rows` |
| many chemical potentials at once | one column each; `wn_to_l` takes them together |
