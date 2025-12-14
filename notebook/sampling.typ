= Sparse sampling notes (IR + Matsubara): kernels, symmetry, weights, and CPQR

This note consolidates a practical strategy for selecting sampling points when using the IR basis, with an emphasis on

- a *dimensionless* kernel definition,
- Matsubara sampling with *± pairs* and *no complex arithmetic* (via real/imag stacking),
- centrosymmetric/symmetric constraints (pair selection),
- nonuniform candidate meshes (log-like) + quadrature-style weights, and
- a simple practical rule for the number of samples: about *IR size + 20%*.

== 1. Dimensionless variables (the IR way)

Let

- $Lambda = beta omega_("max")$ (the only parameter that matters for the IR basis),
- $t = tau / beta in [0, 1]$,
- $x = omega / omega_("max") in [-1, 1]$.

Continuous-time (logistic) kernel for fermions can be written as

$ K^L(t, x; Lambda) = exp(-Lambda t x) / (1 + exp(-Lambda x)) $.

So the continuous kernel depends only on $(t, x)$ and $Lambda$.

== 2. Matsubara kernel in dimensionless form

For Matsubara frequencies,

- fermion: $nu_n = (2n+1) pi / beta$,
- boson:   $nu_n = 2n pi / beta$.

Define the *dimensionless Matsubara variable*

$ u_n = nu_n / omega_("max") $.

Then

- fermion: $u_n = (2n+1) pi / Lambda$,
- boson:   $u_n = 2n pi / Lambda$.

The (Lehmann) Matsubara kernel is

$ K(i nu, omega) = 1 / (i nu - omega) $.

In dimensionless variables,

$ K(i nu_n, omega) = (1 / omega_("max")) * 1 / (i u_n - x) $.

So for *kernel evaluation* you can implement the dimensionless core

- `K(u, x) = 1 / (i*u - x)`

and treat the prefactor `1/omega_max` separately if needed.

=== 2.1 Avoiding complex numbers: explicit real/imag parts

Write

$ 1/(i u - x) = (-x - i u) / (x^2 + u^2) $.

Therefore

- `Re K(u, x) = -x / (x^2 + u^2)`  (even in $u$)
- `Im K(u, x) = -u / (x^2 + u^2)`  (odd in $u$)

and the conjugacy relation

$ K(-u, x) = overline(K(u, x)) $

is automatic.

Practical implication:

- you can select only *positive* Matsubara points (representatives),
- and ensure the method “sees” both real+imag information by *stacking* real/imag blocks.

== 3. CPQR column selection (and how to get rows)

CPQR (QR with column pivoting) is a standard greedy way to pick “important columns”.

- For a real matrix $A$, CPQR returns a pivot order of columns.
- The first $k$ pivots are used as a column index set.

To pick rows, apply the same idea to $A^T$.

In pseudocode:

```python
# column sampling
J = col_piv(A, k)

# row sampling
I = col_piv(A.T, k)
```

== 4. Matsubara sampling: ± pairing + real stacking

Goal: pick a set of Matsubara points that is symmetric under $nu -> -nu$.

=== 4.1 Representative set (positive Matsubara)

Pick representatives among positive indices (e.g. $n = 0, 1, 2, ...$ for fermions).
The final set will be expanded to ± pairs.

=== 4.2 Build a real matrix for selection (no complex arithmetic)

Suppose your selection matrix is built from the Matsubara kernel evaluated at

- candidate Matsubara points $u_n$ (rows), and
- candidate $x$ points (columns),

or the transpose depending on what you want to select.

To avoid complex arithmetic, use

- `K_even = Re(K)`
- `K_odd  = Im(K)`

and run CPQR once on the stacked real matrix

- `A = vcat(alpha * K_even, beta * K_odd)`.

This replaces “pick pivots for even and odd separately and then merge” by a single, consistent objective.

== 5. Enforcing symmetry / centrosymmetry by construction

Sometimes you want the selected *columns* (or rows) to be symmetric under index reversal.
This includes

- $x -> -x$ (frequency symmetry on a symmetric grid),
- $tau -> beta - tau$ (time symmetry),
- or more generally centrosymmetry of a discretized matrix.

Let `mirror[j]` be the index symmetry map (an involution): `mirror[mirror[j]] == j`.

=== 5.1 Symmetric closure (simple but changes k)

Given a set `S`, symmetric closure is

- `S <- S ∪ mirror(S)`.

This is easy but may change the number of points (often forces even counts).

=== 5.2 Pair selection (recommended when you want exactly k points)

Select *pairs* as atomic units.

Let $K$ be the real matrix you want to sample columns from.
Define a representative set $R$ containing one index from each mirror pair (e.g. the “positive side”).

For each representative $r in R$, define a *pair column*

$ a_r = [ K[:, r] ; K[:, m(r)] ] $,

and form the pair matrix

$ A_("pair") = [ a_r ]_(r in R) $.

Run CPQR on `A_pair` to pick `k_rep` representatives, then expand to full symmetric columns

- `k = 2*k_rep` (except for fixed points where `m(r)=r`).

This guarantees symmetry by construction and avoids “add symmetry at the end”.

== 6. Weights (nonuniform meshes and quadrature-style measures)

If you generate candidates on a nonuniform mesh (Gauss–Legendre segments, log-like spacing, adaptive refinement), you typically need weights.

A clean way is to scale the matrix so that CPQR respects a weighted inner product.

Let $w_r$ be the row weights and $w_c$ be the column weights.
Let $D_r$ be a diagonal matrix with diagonal entries $sqrt(w_r)$, and let $D_c$ be a diagonal matrix with diagonal entries $sqrt(w_c)$.
Then define

$ K_w = D_r * K * D_c $.

Then run CPQR on $K_w$.

=== 6.1 Cell-width weights from neighbor distances

Given a sorted 1D grid $x_0 < x_1 < ... < x_(N-1)$, a robust “cell width” weight is

- interior: $w_i = (x_(i+1) - x_(i-1)) / 2$
- endpoints: $w_0 = x_1 - x_0$, $w_(N-1) = x_(N-1) - x_(N-2)$.

This works naturally for log-like meshes.

=== 6.2 The “zero weight at 0” pitfall

If a designed weight function makes $w(0)=0$, scaling by `sqrt(w)` can erase the contribution of the 0-point.
Practical fixes:

- treat 0 as a *special fixed point* (always include it), or
- floor the weight: `w <- max(w, eps)`.

For fermionic Matsubara there is no $nu=0$, but for bosons there is.

== 7. Candidate meshes: log-like + refine + re-select

Uniform grids are often wasteful.
A practical approach is:

- build a *log-like* candidate mesh (dense near difficult regions),
- assign weights via cell widths,
- run selection,
- optionally refine the mesh around selected points and run selection again.

For symmetric domains, generate candidates in pairs (±) to keep symmetry straightforward.

== 8. How many samples? (IR size + 20%)

When sampling to recover IR coefficients of size $L$, a pragmatic default is

- `k = ceil(1.2 * L)`.

If you enforce symmetry by pair selection, round to an even number (or choose representatives)

- `k_rep = ceil(0.6 * L)` and expand to `k ≈ 2*k_rep`.

This is not a theorem, but in practice it often gives stable least-squares recovery without overcomplicating tuning.

== 9. Minimal implementation sketch (dimensionless Matsubara kernel)

```python
import numpy as np

def matsubara_u(n, Lambda, stat="F"):
    if stat == "F":
        return (2*n + 1) * np.pi / Lambda
    else:
        return (2*n) * np.pi / Lambda

def K_re(u, x):
    return -x / (x*x + u*u)

def K_im(u, x):
    return -u / (x*x + u*u)

# build selection matrix on a candidate grid
# U: array of u_n (positive reps)
# X: array of x in [-1,1]
# K_even = Re, K_odd = Im, then stack to a real matrix for CPQR.
```

(For Rust, the same formula works in `f64` without complex types.)
