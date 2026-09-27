# Conventions

Three conventions are worth stating before any of the tutorials, because they
are the ones that most often differ between codes.

## Imaginary time runs over `[0, β)` — but the sampling times do not

Green's functions are functions of `τ ∈ [0, β)`, and that is the domain the
basis functions `uₗ(τ)` are defined on.

The *sampling* times, however, are reported on the symmetric interval
`[−β/2, β/2]`. `sparse-ir` picks them as the roots of the first discarded basis
function and then folds each one that landed beyond `β/2` down by `β`, which
for a fermionic function means flipping its sign as well:

```text
    G(τ − β) = −G(τ)   (fermionic),      G(τ − β) = +G(τ)   (bosonic).
```

The folding is not cosmetic. Near `τ = β` the values of `G` are tiny and their
relative accuracy is poor; the reflected point near `τ = 0` carries the same
information with far more significant digits. Everything that takes a `τ`
accepts the whole range `[−β, β]` and applies the relation above, so you can
hand it either form — but if you compare sampling points against another
library, compare them after folding.

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

## Nothing is computed twice by accident

Constructing a `FiniteTempBasis` runs a singular value expansion, which is by
far the most expensive thing in the library. It depends on the kernel and the
accuracy only, never on `β` — so if you need a fermionic and a bosonic basis
for the same kernel, compute the expansion once and build both bases from it
with `FiniteTempBasis::from_sve_result`.
