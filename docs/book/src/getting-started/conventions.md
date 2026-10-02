# Conventions

This page fixes the notation and the conventions used throughout the book.
They are the ones that most often differ between codes, so read it before
comparing numbers with another library.

## Notation

| Symbol | Meaning |
|---|---|
| \\(\beta\\), \\(\omega_\mathrm{max}\\), \\(\Lambda = \beta\omega_\mathrm{max}\\) | inverse temperature, frequency cutoff, dimensionless cutoff |
| \\(\tau\\) | imaginary time, \\(\tau \in [0, \beta)\\) |
| \\(\omega\\) | real frequency |
| \\(\mathrm{i}\nu_n = \mathrm{i}n\pi/\beta\\) | Matsubara frequency with the *reduced* index \\(n\\) (see below) |
| \\(\mathrm{i}\nu\\), \\(\mathrm{i}\omega\\) | fermionic and bosonic Matsubara frequencies, when one formula has both |
| \\(m\\) | textbook Matsubara index, \\(n = 2m + \zeta\\) |
| \\(\zeta\\) | \\(1\\) for fermions, \\(0\\) for bosons |
| \\(u_l(\tau)\\), \\(\hat u_l(\mathrm{i}\nu)\\), \\(v_l(\omega)\\), \\(s_l\\) | IR basis functions and singular values, \\(l = 0, \dots, L-1\\) (`basis.u`, `basis.uhat`, `basis.v`, `basis.s`) |
| \\(g_l\\) | IR expansion coefficients |
| \\(\omega_p\\), \\(c_p\\) | DLR poles and coefficients |
| \\(\varepsilon\\) | accuracy parameter of a representation (see below) |

## Three representations, one accuracy parameter

`sparse-ir` provides three compact representations of a Green's function. Each
is built from \\(\beta\\), \\(\omega_\mathrm{max}\\) and an accuracy
\\(\varepsilon\\), but \\(\varepsilon\\) means something different for each:

| Representation | Built by | \\(\varepsilon\\) means |
|---|---|---|
| IR basis ([sparse sampling](../tutorials/sparse_sampling.md)) | `FiniteTempBasis::new(LogisticKernel::new(beta * wmax)?, beta, Some(eps), None)` | singular-value cutoff: keep \\(l\\) with \\(s_l/s_0 > \varepsilon\\) |
| DLR ([discrete Lehmann representation](../tutorials/dlr.md)) | `DiscreteLehmannRepresentation::new(beta, wmax, eps)` | pivot tolerance of the interpolative decomposition that selects the poles |
| MiniPole ([MiniPole](../tutorials/minipole.md)) | `mini_pole_dlr_from(&dlr, &coeffs, &params)` | ESPRIT tolerance `err`; it controls how many poles are kept and is **not** a bound on the reconstruction error |

The IR basis and the DLR are fixed, data-independent representations: build
them once for given \\(\beta\\), \\(\omega_\mathrm{max}\\) and
\\(\varepsilon\\), and expand any Green's function in them. MiniPole is
data-adapted: it compresses one particular function into a few poles, which may
lie off the real axis.

## Imaginary time runs over `[0, β)`, but the sampling times do not

Green's functions are functions of \\(\tau \in [0, \beta)\\), and that is the
domain the basis functions \\(u_l(\tau)\\) are defined on.

The *sampling* times, however, are reported on the symmetric interval
\\([-\beta/2, \beta/2]\\). `sparse-ir` picks them as the roots of the first
discarded basis function \\(u_L\\) and then folds each one that landed beyond
\\(\beta/2\\) down by \\(\beta\\), which for a fermionic function means flipping
its sign as well:

```text
    G(τ − β) = −G(τ)   (fermionic),      G(τ − β) = +G(τ)   (bosonic).
```

The folding is not cosmetic. Near \\(\tau = \beta\\) the values of \\(G\\) are
tiny and their relative accuracy is poor; the reflected point near
\\(\tau = 0\\) carries the same information with far more significant digits.
Everything that takes a \\(\tau\\) accepts the whole range \\([-\beta, \beta]\\)
and applies the relation above, so you can hand it either form. If you compare
sampling points with another library, compare them after folding.

## Statistics is part of the type

A basis is fermionic or bosonic, and which one it is shows up in the type, not
in a runtime flag:

```rust
use sparse_ir::{Fermionic, FiniteTempBasis, LogisticKernel};

let (beta, wmax, eps) = (10.0, 1.0, 1e-10);
let kernel = LogisticKernel::new(beta * wmax).unwrap();
let basis =
    FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(eps), None).unwrap();
assert!(basis.size() > 0);
```

## Matsubara frequencies: the reduced index

A Matsubara frequency is stored as one integer \\(n\\), the `MatsubaraFreq`
type (`FermionicFreq`, `BosonicFreq`), with

\\[
  \nu_n = \frac{n\pi}{\beta}, \qquad
  n \text{ odd for fermions } (\pm1, \pm3, \dots), \quad
  n \text{ even for bosons } (0, \pm2, \dots).
\\]

This \\(n\\) is the *reduced* index. It is **not** the textbook index \\(m\\)
of \\(\nu = (2m + \zeta)\pi/\beta\\); the two are related by
\\(n = 2m + \zeta\\). The lowest positive fermionic frequency \\(\pi/\beta\\)
is therefore `FermionicFreq::new(1)`, and a fermionic `n = 0` or a bosonic
`n = 1` is rejected:

```rust
use sparse_ir::{BosonicFreq, FermionicFreq};
use std::f64::consts::PI;

let beta = 10.0;
assert!((FermionicFreq::new(1).unwrap().value(beta) - PI / beta).abs() < 1e-15);
assert!((BosonicFreq::new(2).unwrap().value(beta) - 2.0 * PI / beta).abs() < 1e-15);
assert!(FermionicFreq::new(0).is_err());
assert!(BosonicFreq::new(1).is_err());
```

Every integer the library hands you, such as the points returned by
`MatsubaraSampling::sampling_points()`, is a reduced index. If you index an
array by \\(m = 0, 1, 2, \dots\\), convert with \\(m = (n - \zeta)/2\\).

One place uses a different convention. The MiniPole entry points that start
from a DLR (`mini_pole_dlr_from`, `mini_pole_dlr`) evaluate the function on the
contour \\(\omega_n = (2n+1)\pi/\beta\\) with a *contour* index \\(n\\), and
they do so even for a bosonic DLR. The [MiniPole](../tutorials/minipole.md)
page explains why.

## Nothing is computed twice by accident

Constructing a `FiniteTempBasis` runs a singular value expansion (SVE), which
is by far the most expensive step in the library. The SVE depends only on the
kernel (through \\(\Lambda\\)) and on \\(\varepsilon\\). If you need a
fermionic and a bosonic basis for the same \\(\Lambda\\), compute the expansion
once and build both bases from it with `FiniteTempBasis::from_sve_result`:

```rust
use sparse_ir::{compute_sve, Bosonic, Fermionic, FiniteTempBasis, LogisticKernel, TworkType};

let (beta, wmax, eps) = (10.0, 1.0, 1e-10);
let kernel = LogisticKernel::new(beta * wmax).unwrap();
let sve = compute_sve(kernel, Some(eps), None, None, TworkType::Auto).unwrap();
let fermionic = FiniteTempBasis::<LogisticKernel, Fermionic>::from_sve_result(
    kernel, beta, sve.clone(), Some(eps), None,
)
.unwrap();
let bosonic =
    FiniteTempBasis::<LogisticKernel, Bosonic>::from_sve_result(kernel, beta, sve, Some(eps), None)
        .unwrap();
assert_eq!(fermionic.size(), bosonic.size());
```

The two bases share their singular values and functions; only the statistics
of \\(\hat u_l(\mathrm{i}\nu)\\) differs.
