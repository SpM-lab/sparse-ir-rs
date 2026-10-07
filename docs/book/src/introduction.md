# Introduction

`sparse-ir` compresses the imaginary-time Green's functions of finite-temperature
many-body theory. A Green's function on a dense grid in imaginary time or
Matsubara frequency carries far fewer independent numbers than grid points, and
the library provides three ways to exploit that:

- **The intermediate representation (IR) basis.** An orthonormal basis
  obtained from a singular value expansion of the analytic-continuation kernel.
  Its size grows only logarithmically with the cutoff
  \\(\Lambda = \beta\omega_\mathrm{max}\\), and a matching set of *sparse
  sampling* points in \\(\tau\\) and in Matsubara frequency is enough to
  recover the expansion. See [Sparse sampling](tutorials/sparse_sampling.md)
  and [Transformation from and to IR](tutorials/transformation.md).
- **The discrete Lehmann representation (DLR).** A sum of simple poles on a
  fixed real-frequency grid, chosen by an interpolative decomposition of the
  kernel. It needs no IR basis, comes with its own sampling nodes, and can also
  be built from an existing IR basis. See
  [Discrete Lehmann representation](tutorials/dlr.md).
- **MiniPole.** A data-adapted compression: ESPRIT extracts a handful of
  complex poles from one particular function, which also gives an analytic
  continuation. See [MiniPole](tutorials/minipole.md).

The IR basis and the DLR are fixed representations shared by every function
with the same \\(\beta\\), \\(\omega_\mathrm{max}\\) and accuracy; MiniPole
adapts to the data. For how the three are related, where each came from and
what each is convenient for, see
[IR, DLR and MiniPole: history and comparison](https://spm-lab.github.io/sparse-ir-doc/src/history_comparison.html)
on the theory site.

## How this book is organised

*Getting started* covers installation and the conventions used throughout,
including the Matsubara-frequency index; read
[Conventions](getting-started/conventions.md) before comparing numbers with
another code. *Representations* introduces the IR basis, the DLR and MiniPole.
*Analytic continuation* goes from imaginary-time data back to real frequency.
*Applied examples* solve complete many-body problems with the IR basis.

Most pages follow a notebook of the Python and Julia tutorials,
[sparse-ir-tutorial-v2](https://spm-lab.github.io/sparse-ir-tutorial-v2/);
each page says which one.

## How this book is built

The Rust code on these pages is included from the example programs in
`docs/tutorial-code/` rather than copied by hand, so it is the code that is
actually compiled and run, and every figure was drawn from numbers those
programs wrote. Most examples are checked against reference values from the
Python implementation; the rest, such as the IR-independent DLR and MiniPole,
check themselves against closed-form results. Each page ends with the command that reproduces it; run
it from `docs/tutorial-code`.
