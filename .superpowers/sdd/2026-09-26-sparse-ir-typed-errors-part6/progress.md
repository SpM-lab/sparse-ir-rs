# 部分 6 実行台帳

計画: `docs/superpowers/plans/2026-09-26-sparse-ir-typed-errors-part6.md`
起点: 32b0e73（部分 5 の最後）
作業ツリー: `/Users/hiroshi/projects/sparse-ir/worktrees/wt-te`（新しいマシン、2026-09-27 に再開）
ログ: `/Users/hiroshi/projects/sparse-ir/worktrees/logs/part6`

## 環境の差（新しいマシン、2026-09-27）

- `$WS = /Users/hiroshi/projects/sparse-ir`（HANDOFF §−1 の読み替え）。旧 `worktrees/wt-te`・`wt-base` は同じ相対位置に作り直した。
- OS は macOS（arm64）。元のマシンは Linux。
- **Ruling（警告の基準の比べ方）:** `$H/part2-base-warn.txt` との diff は、そのままでは 3 点で必ず食い違う。どれも警告の中身ではない。
  1. `uniq -c` の桁詰めが GNU と BSD で違う（`      1` と `   1`）。
  2. BLAS の backend の行（Linux: `Found system BLAS: openblas`、macOS: `Using macOS Accelerate framework for BLAS`）。
  3. 同じ件数の行どうしの `sort` の順序が locale で違う。
  そこで比較は「先頭の空白を潰す → BLAS の行を除く → `LC_ALL=C sort`」で正規化して行う。正規化した集合が基準と一致することを、各タスクで確かめる。

## Task 1: 部分 4、5 の後回しの Minor（文書と文言） — 完了

- Step 1〜4 は元のマシンの未コミットの変更（`part6-task1-wip.patch`）。`git apply` して内容を計画と突き合わせ、Step 1（`new` の `# Errors` に `NotSupported`）、Step 2（重複した極の文、コアと capi の doc）、Step 3（`test_complex_to_real_zz_checks_the_input_first` の doc comment）、Step 4（`unsupported` の文言）がそのとおり入っていることを確かめた。
- `git grep 'for this sampling type'` は空。
- Step 5: cbindgen 0.29.2 でヘッダを作り直した。`include/sparseir/sparseir.h` の差分はコメントだけ（非コメント行の差分なし）。`assets/sparse_ir_capi.h` も同じ。WIP のヘッダの差分と一致した。
- Step 6: `sparse-ir --lib` 398 passed / 0 failed。`sparse-ir-capi` 171 passed（1+11+2+3+36+40+78）/ 0 failed。ビルドの警告は正規化して基準と一致。
- Step 7: コミット。

## Task 2: `spir_sve_result_from_matrix{,_centrosymmetric}` の nx / ny の検査 — 実行中

- Step 1: 呼び出し元は計画の Expected どおり。C++ は `cinterface_core.cxx` の TEST_CASE 1 つ、Fortran と Python にはない。
- Step 2〜3: 新しいテスト `sve_result_from_matrix_rejects_sizes_other_than_the_gauss_points` を `tests/test_invalid_input_status.rs` に足した。RED は Expected どおり（最初の失敗は `from_matrix, nx = 5, ny = 4` で -7、`sve/utils.rs` の範囲外の添字の panic）。
- Step 4: `is_gauss_point_count` を足し、非中心対称は `validated` の match の直後、中心対称は `segs[0] == 0` の検査の直後に置いた。
- Step 5: 2 つの関数の doc に `nx` / `ny` の条件と `SPIR_INVALID_ARGUMENT` の行を足した。
- Step 6 と **Ruling（epsilon のテストも直した）**:
  - 計画は `test_sve_result_from_matrix_centrosymmetric_requires_segments_from_zero` だけを直すとしていたが、同じファイルの `test_sve_result_from_matrix_centrosymmetric_rejects_epsilon_before_the_svd` も同じ `m.nx + 1` の行列を使っていた。新しい大きさの検査があると、この入力は大きさの誤りとしても弾かれ、テストが epsilon の検査を見分けなくなる。そこで同じように `m.nx` の正しい行列に直した（期待値 -6 は変えない）。
  - Global Constraints は「既存テストの入力を変えるのは 3 つだけ、ほかは止めて報告」としている。status は 1 つも変わらないが、計画の範囲外の変更なので、ここに Ruling として残し、最後の報告でも挙げる。
- **Ruling（事前検査を外したときの RED の実測）:** 計画の任意の確認を行った（どちらも一時的に外して測り、元に戻した。コミットしていない）。
  - 中心対称の `segs_x_slice[0] != 0.0 || segs_y_slice[0] != 0.0` を外しても、`..._requires_segments_from_zero` は**通る**。正しい大きさの行列では、コアが SVD の後に同じ -6 を返すため。したがってこのテストは事前検査そのものの見張りにはならない。事前検査は決めたこと 5・14（コアの検査は SVD の後＝重い計算の後）により、D8 でも残す（表の K5）。
  - `epsilon` の検査（2 つの関数）を外しても `..._rejects_epsilon_before_the_svd` は通る。コアが epsilon を同じ -6 で弾くため。これも事前検査そのものの見張りではない。どちらも Task 5 の D8 の判断に効く実測なので記録する。
- Step 7: capi の全テスト緑（1+11+2+3+36+41+78 = 172 passed、0 failed）。新しいテストの分だけ 171 → 172 に増えた。
- Step 8: C++ の `Test spir_sve_result_from_matrix` を直した（コメント、`nx = n_gauss * n_segments_x`、Gauss 則の区間の数、`if (status == ...)` → `REQUIRE`、行優先と列優先の両方）。`run_with_rust_capi.sh` は緑: `All tests passed (41788 assertions in 2 test cases)` と `(351 assertions in 13 test cases)`。部分 2 のゲートの 347 から 4 増えたのは、新しい `REQUIRE` 4 つ。決めたこと 9 の probe のとおり、直した大きさで成功した。
- Step 9: ヘッダを作り直した。非コメント行の差分なし。
- **Ruling（C++ の後は `cargo clean` してから警告を比べる）:** `run_with_rust_capi.sh` は作業ツリーの `target` を消してから release でビルドし直す。その後に `cargo build --workspace --all-targets`（debug）を走らせると、macOS の linker が `warning: (arm64) deps/....rcgu.o unable to open object file` を 40 種類ほど出す。コードとは関係のない、消された中間ファイルの debug 情報への参照である。`cargo clean` してからビルドし直すと消え、警告の集合は基準と一致した。以後、C++ を走らせたタスクでは `cargo clean` してから比べる。
- Step 10: コミット 6812178。

## Task 3: capi の `PolyVectorFuncs` と評価関数を `Result` にする、正則化関数の定義域 — 未着手

## Task 3 — done (6a0dc5c)

Steps 1-10 as planned. `PolyVectorFuncs::{evaluate_at,batch_evaluate_at}` and
the four `spir_funcs` eval helpers return `Result` / `Option<Result<..>>`; the
four C entry points map the error with `status_from`;
`spir_basis_new_from_sve_and_regularizer` returns the error of its test
evaluation instead of panicking. Header diff is comment-only (the
`SPIR_INVALID_ARGUMENT` list of that function); `assets/sparse_ir_capi.h`
synced. `cargo test -p sparse-ir-capi --release`: 0 failed, both new tests pass.

Ruling: the warning baseline needed one more filter. On this machine the
linker emits ~500 `(arm64) ... rcgu.o unable to open object file` warnings when
linking debug test binaries. The earlier note that `cargo clean` removes them is
wrong — measured twice: clean + build emits them, and the next build replays the
identical set from cargo's cache. With them filtered out the warning set is
identical to `part2-base-warn.txt`. `agent-skills/build-warning-baseline` was
corrected in the same commit.

Also done, at the user's request (dfccfda): `cxx_tests/run_with_rust_capi.sh
--no-clean` (measured: 58 s incremental, all tests pass), and two repo skills,
`local-test-gate` and `build-warning-baseline`, registered in AGENTS.md.

## Task 4 — done (755b859, cd79966)

Step 1's bit-for-bit guard passed on the old code, as the plan predicted, and
still passes after the rewrite: the C DLR functions now hold a `DlrOf` and the
indices of the selected poles and read their values from the core's
`evaluate_tau` / `evaluate_matsubara`. Step 4's grep is empty — no `zero_pole`,
`pole_weight`, `kernel_ypower` or `unwrap_or_else(|e| panic!` left in
`sparse-ir-capi/src`. Full capi suite: 0 failed. Header unchanged (no C doc
changed). Warnings equal BASE.

Deviation: the plan wrote `Complex64` and `StatisticsType` as if imported;
`types.rs` imports `num_complex::Complex`, so `DlrOf::matsubara_values` returns
`Vec<Vec<Complex<f64>>>` (the same type) and `StatisticsType` was added to the
`sparse_ir::traits` import. `sparse_ir` re-exports `DTensor`, but as the dynamic
`DTensor<T, DynRank>`; `columns` takes `mdarray::DTensor<T, 2>` as the plan
wrote it.

## Task 5: D8 pre-check tidy-up

### Step 1 — locate the pinning tests (done)

Baselines before any change: `test_invalid_input_status` 42 passed,
`test_basis_edge_cases` 11 passed, lib filter
`gauss piecewise kernel sve_result_truncate` 19 passed.

All named tests were found:

- 9 in `sparse-ir-capi/tests/test_invalid_input_status.rs`
- `default_matsus_of_a_basis_without_parity_are_not_supported` in
  `sparse-ir-capi/tests/test_basis_edge_cases.rs`
- `test_gauss_legendre_rule_piecewise_double` in `sparse-ir-capi/src/utils.rs`
- `test_dlr_transforms_reject_a_wrong_target_extent` and
  `test_ir2dlr_rejects_an_output_too_large_to_address` in
  `sparse-ir-capi/src/dlr.rs`

**Finding (recorded per the plan's "expected vs. actual" rule):**
`test_gauss_legendre_rule_piecewise_double` exercises only the SUCCESS cases
(`[-1, 1]` and `[-1, 0, 1]`). It does not pin the rejection of non-monotonic
boundaries, so D8-4's `SPIR_INVALID_ARGUMENT` path was unpinned before this
task. The Step 2 test now pins it, for both the `double` and the `ddouble`
entry point.

### Step 2 — guard test (done)

Added `constructor_scalars_checked_by_the_core_keep_their_statuses` to
`sparse-ir-capi/tests/test_invalid_input_status.rs`. It PASSES on the current
code, as the plan expected: every scalar listed in the plan already yields
`SPIR_INVALID_ARGUMENT` through the pre-checks. No status differs from the
table or from survey Appendix B.

Deviation: the plan wrote only the `double` variant of the piecewise
Gauss-Legendre rule. The `ddouble` entry point
(`spir_gauss_legendre_rule_piecewise_ddouble`, 8 arguments) has the same
pre-check and was unpinned as well, so the test covers both.

### Steps 3-8 — removing the pre-checks (done)

D8-1 … D8-10 removed as the table says; D8-11 split into
`utils::transform_dims`, which the four DLR transforms now call with the
`orig_dims` that `validate_dims` has already returned (one validation instead
of two). The helpers that fell dead were removed as well:
`funcs::is_matsubara_index`, `types::{tau_domain, is_in_domain,
has_definite_parity}`, `spir_basis::has_default_matsubara_sampling_points`,
`spir_funcs::{continuous_domain, matsubara_statistics}` and `DlrOf::statistics`
(the last is from Task 4; it was only read by `matsubara_statistics`).

Step 6 GREEN: every capi test binary passes, 0 failed — 43 in
`test_invalid_input_status` (42 + the Step 2 test), 11 in
`test_basis_edge_cases`, 78 in the lib, 37 + 3 + 2 + 1 in the rest. No status
changed, so no test had to be reverted to a "keep" ruling.

Step 7: the only `(#266)` references left are on checks that stay — the
`order` check of the two batch entry points (K1) and the τ finiteness of
`spir_tau_sampling_new_with_matrix` (K8). No helper name survives outside the
test `tau_sampling_new_with_matrix_does_not_check_the_tau_domain`.

Step 8: the header diff is comment-only (21 added lines, the DLR sentences of
Step 5); `assets/sparse_ir_capi.h` re-synced. The warning set after
`cargo clean && cargo build --workspace --all-targets` equals
`part2-base-warn.txt` exactly under the normalizer of the
`build-warning-baseline` skill.

## Task 6: the statuses of invalid input in C++, Fortran and Python

Step 1 (C++): `cxx_tests/cinterface_core.cxx` gained 3 static helpers
(`stand_in_kernel_matrix`, `stand_in_sve`, `fermionic_basis`) and 12
`TEST_CASE(..., "[status]")` blocks, S1-S12. The suite grew 13 -> 25 test
cases, 351 -> 417 assertions, all green.

Step 2 (Fortran): new `fortran/test/test_status_codes.f90`
(`program test_status_codes`, helpers `check`, `check_null`,
`stand_in_kernel_matrix`, `fermionic_basis`, subroutines S1-S12).
Auto-discovered by the test CMakeLists glob; 11/11 -> 12/12 programs, green.

Step 3 (Python): new `python/tests/test_status_codes.py` with one test per
case, S1-S12 (12 tests). `SPIR_TWORK_AUTO` is not exported by
`pylibsparseir.constants`, so the literal -1 is used. NULL out-pointers are
checked with `not ptr`. `python/tests/c_api/integration_tests.py`: the
DLR->IR wrong-dims case now fills `ir_coeffs` with a sentinel and asserts
both `SPIR_INPUT_DIMENSION_MISMATCH` and that nothing was written.

Step 4: `uv run pytest tests/test_status_codes.py tests/c_api/integration_tests.py -q`
-> 24 passed (12 new + 12 integration).

Step 5: commit 04b99ad "Pin the statuses of invalid input in the C++,
Fortran and Python tests". `fortran/_build_gfortran/` left untracked.

## Task 7: verification, stale wording, warnings, overall check

Step 1: greps 2 and 3 of the plan were not empty/empty as expected:
- grep 1 found three comments naming "part 7" in `sparse-ir`
  (`sve/strategy.rs:476`, `sve/types.rs:155`, `sve/utils.rs:446`). The word
  was removed from all three; the items themselves are already on part 7's
  list (private `SVEResult` fields, `safe_epsilon` -> pub(crate)).
- grep 3 found one stale comment, in the test module of
  `sparse-ir-capi/src/sve.rs:1362`: it claimed `n_segments_x/_y` were
  "the number of boundary points (n_segments + 1)", which is backwards -
  they are the number of segments, and the boundary arrays hold one more
  entry. Reworded; the two trailing "n_segments_x - 1 is the number of
  segments" comments on the `nx`/`ny` lines were wrong for the same reason
  and were dropped.
- grep 4 listed exactly the invariants of decision 20 and nothing else:
  `basis.rs` 504/519/525 `flush().unwrap()`, `dlr.rs` 99/219
  `unreachable!()`, `sampling.rs` 480/696 `assert_eq!`, `types.rs:863`
  `partial_cmp(..).unwrap()`, `utils.rs` 110/141 `assert!` and 468
  `.expect`.

Step 2: `header sync OK` (cbindgen == include/ == assets/). The diff of
`include/sparseir/sparseir.h` against 32b0e73 has no non-comment line: the
header changed by 21 added comment lines only, so the C ABI is unchanged.

Step 3: build warnings identical to `part2-base-warn.txt` (after a
`cargo clean`; the normalizing filter drops the BLAS/object-file lines).
Doc warnings identical to `ea718e3-doc-warn.txt` - no line in either
direction, so no intra-doc link was left pointing at a deleted function.

Step 4 gate (p6, logs in worktrees/logs/p6-*.log): FMT OK; build 0;
test 591 passed / 0 failed / 5 ignored; doctest 19; system-blas 416 / 5
ignored; sysblas doctest 19; header sync OK; cxx 41788 + 417 assertions
(2 + 25 test cases); Fortran 12/12; Python 97 passed; check_version see
below.

Two numbers differ from the plan's expectation by one, in both cases
because the part 5 ledger's figures were one too high:
- the core under system-blas is 416, not 417: `gemm.rs`
  `test_default_backend_is_faer` is `#[cfg(not(feature = "system-blas"))]`,
  a pre-existing gate, so 417 is the core count *without* the feature.
- the total is 591, not 588 + 4 = 592, i.e. part 5 ended at 587. Verified
  against git rather than against the ledger: between 32b0e73 and HEAD the
  `#[test]` count of `sparse-ir` is unchanged at 423 and the only diff in
  `sparse-ir/` is two doc comments, while `sparse-ir-capi` gained exactly
  the 4 expected test functions and lost none.

`check_version.py` exits 1 under the macOS system `python3` (3.9.6), which
cannot parse the `str | None` annotation in `extract_workspace_version`.
It is unrelated to this work (the script last changed in a8075c7 and no
version metadata was touched); under `python3.12` it exits 0 with only the
known Julia 0.8.4 != 0.9.0 warning.

Step 5: commit f1b652b "Update the documentation of the C checks" (the four
comment fixes of Step 1; comments only, no code). `cargo fmt --all` changed
nothing further; the capi test binaries were re-run green afterwards.

Step 6: not pushed. C API status changes of part 6: none beyond the plan's
table. Appendix B holds. Ruling: none needed.

### Carried to part 7
- `SVEResult`'s public fields (`s` and the poly vectors) become private;
  `sve/strategy.rs` and `sve/utils.rs` build `PiecewiseLegendrePolyVector`
  through the public field today.
- `safe_epsilon` becomes pub(crate); its panic on a negative or NaN
  `eps_required` is then unreachable from outside the crate
  (`sve/types.rs` tests).
