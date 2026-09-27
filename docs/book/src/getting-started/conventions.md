# Conventions

Two conventions are worth stating before any of the tutorials, because they are
the two that most often differ between codes.

## Imaginary time runs over `[0, β)`

Green's functions are functions of `τ ∈ [0, β)`, and that is the domain
`sparse-ir` samples on and expands in.

The basis functions themselves, however, are defined on the *symmetric*
interval, which is why the dimensionless variable `x = 2τ/β − 1` runs over
`[−1, 1]`. Sampling points you get back from the library are always in `τ`, not
in `x`.

## Statistics is part of the type

A basis is fermionic or bosonic, and which one it is shows up in the type, not
in a runtime flag:

```rust,ignore
use sparse_ir::{Bosonic, Fermionic, FiniteTempBasis};
```

The Matsubara frequencies follow: `ωₙ = (2n+1)π/β` for fermions and `2nπ/β` for
bosons. `sparse-ir` indexes them by the integer `n` — the `MatsubaraFreq` type —
rather than by the frequency itself, so that the two statistics can share one
representation without rounding ever confusing an even index for an odd one.
