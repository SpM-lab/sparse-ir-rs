# Introduction

`sparse-ir` compresses the imaginary-time Green's functions of finite-temperature
many-body theory. Instead of a dense grid in imaginary time or Matsubara
frequency, it gives you a basis whose size grows only logarithmically with the
cutoff `Λ = β ω_max`, together with the handful of sampling points at which you
need to know a function to recover its expansion.

For how the IR basis, the discrete Lehmann representation (DLR) and MiniPole
are related, where each came from and what each is convenient for, see
[IR, DLR and MiniPole: history and comparison](https://spm-lab.github.io/sparse-ir-doc/src/history_comparison.html) on the theory site.

This book is the tutorial for the Rust implementation. Every code block it
shows is compiled, and every figure it shows was drawn from numbers a program
in `docs/tutorial-code/` actually produced — the same programs the test suite
compares against reference values from the Python implementation. Nothing here
is written out by hand.

The tutorials are ported from the notebooks of
[sparse-ir-tutorial](https://github.com/SpM-lab/sparse-ir-tutorial); each page
says which notebook it came from.
