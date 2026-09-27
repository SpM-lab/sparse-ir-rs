# 部分 7 実行台帳

計画: `docs/superpowers/plans/2026-09-27-sparse-ir-typed-errors-part7.md`
起点: f1b652b（部分 6 の最後）
作業ツリー: `/Users/hiroshi/projects/sparse-ir/worktrees/wt-te`
ログ: `/Users/hiroshi/projects/sparse-ir/worktrees/logs`

警告の比べ方は部分 6 の Ruling（正規化して比べる。BLAS の行だけは
プラットフォームの差として除く）をそのまま使う。

## コミット

| SHA | 内容 | Task |
| --- | --- | --- |
| 46d5067 | Build an SVE from a discretized matrix in the core | 1, 2 |
| 9f9e701 | Hide the fields of the DLR behind accessors | 3 |
| 53ca308 | Hide the fields of SVEResult behind accessors | 4 |
| 89bc569 | Hide the fields of the piecewise Legendre types behind accessors | 5 |
| c8cd930 | Hide the fields of the Gauss rule and the regularized Bose kernel | 6 |
| d520337 | Narrow the public surface of the core | 7 |
| 8c9b1f9 | Correct two comments of the C API | 部分 6 のレビュー |
| 0d898af | Sync the bundled copy of the C header | 8 |
| b76e8c6 | Ignore the build directories the test gate keeps | 8 |

## 計画からの逸脱と、その理由

1. **`Rule` のフィールドをアクセサに機械置換しなかった（Task 6）。** 計画は
   crate 内の約 190 か所を `x()` などに書き換える想定だったが、`pub(crate)`
   にした時点で crate の外からは見えず、結果は同じ。差分を小さく保つため、
   crate 内の読み出しはフィールドのままにした。
2. **`sampling::movedim` は `pub` のまま（Task 7）。** `sparse-ir/examples/roundtrip.rs`
   が別クレートとして使う（126・151・246 行）。
3. **`matrix_from_gauss*` は `pub` のまま（Task 7）。** capi の**テスト**が
   別クレートから使う。
4. **`compute_sve_general` は `pub` のまま（Task 7）。** 中心対称でない
   カーネルの SVE の入口として公開されているべきものだから。（当初ここに
   「テストが使うから」とも書いたが、部分 7 のレビューの指摘どおりそれは
   誤り。利用者は `sve/tests.rs`・`basis_tests.rs`・`polyfourier_tests.rs`
   という crate 内の `#[cfg(test)]` だけで、`pub(crate)` にしてもテストは
   壊れない。残す根拠は設計判断のみ。）`truncate` と `safe_epsilon` だけ `pub(crate)` に
   した（`sve/tests.rs` の `safe_epsilon` の import は `super::types::` 経由に
   変えた）。
5. **DLR の 0 点の極限は `panic!` のまま（Task 3）。** `with_poles` が
   bosonic の 0 の極を他の ypower に対して拒むので、この腕は到達不能。
   到達不能であることを注釈に書いた。
6. **ヘッダが 1 行だけ変わった（Task 8 Step 2 の Expected と違う）。**
   `spir_uhat_get_default_matsus` の doc が呼ぶ Rust の関数名を
   `FiniteTempBasis::default_matsubara_sampling_points_impl` から
   `sparse_ir::basis::default_matsubara_sampling_points_from_uhat` に直した
   ためで、コメントだけ。C の関数・引数・status は変わらない。

7. **`Rule` の順序の検査は足さず、doc だけ書いた（Task 6）。** 計画は
   「順序の事前条件を検査するか、doc に書くか」を求めていた。`Rule::new` /
   `from_vectors` は `pub` で capi の `utils.rs:462` が呼ぶので、ここに検査を
   足すと C の status が増える（部分 7 は status を変えない制約）。よって
   `x()` とフィールドの doc に「non-decreasing order」と書くだけにした。
   `validate()` はテストからのみ呼ばれる。実際の検査を入れるなら 0.11.0 で、
   status の表を伴って。
8. **`from_discretized_matrix` を 2 関数に分けた（Task 1）。** 計画の署名は
   `symmetry: Option<SymmetryType>` を取る 1 関数だったが、非中心対称版と
   `from_discretized_matrices_centrosymmetric` に分けた。後者は even/odd の
   2 つの行列を取るので、引数の形がそもそも違う。`Option` で分岐させるより
   型として素直。

## Task 7（コアの公開面を縮小）の削除

`pub(crate)` 化で crate 内のどこからも使われないと分かったものを、ユーザーの
判断（「削除する」）に従って消した。約 2,100 行、テスト 21 本。

- モジュールごと: `interpolation2d`、`working_buffer`。
- `interpolation1d`: `Interpolate1D`、`interpolate_1d_legendre`、
  `evaluate_interpolated_polynomial`、`evaluate_legendre_basis` ほか。
  残したのは `legendre_collocation_matrix`（SVE が使う）だけ。
- `kernelmatrix`: `InterpolatedKernel`（自分のテストからしか作られていなかった）。
- `col_piv_qr`: `new`、`unpack`、`unpack_r`、`col_piv_qr_internal`、
  `q_tr_mul`、および正方行列の impl 全体（`solve`、`solve_mut`、
  `solve_upper_triangular_mut`、`try_inverse`、`is_invertible`、`determinant`）。
  `rank`・`rank_with_rtol` は早期打ち切りのテストが使うので、この
  ファイルに既にあった `#[allow(dead_code)] // Used in tests` を付けて残した。
- `polyfourier`: `PiecewiseLegendreFT` の**メソッド版**の
  `sign_changes`・`find_extrema` と、そのメソッドからしか呼ばれていなかった
  `func_for_part`、`find_all_roots`、`bisect`、`discrete_extrema`。
  同名の**自由関数**（`polyfourier.rs` の `sign_changes`・`find_extrema`・
  `func_for_part`・`discrete_extrema`）と `PiecewiseLegendrePoly::find_all_roots`
  は健在で、basis の実経路はそちら。
- `poly`: `with_data_and_symmetry`。

テスト数への影響: コアの lib テストは 398 passed / 5 ignored から
**381 passed / 1 ignored** になった（消えた 4 本の ignored は
`gauss_tests.rs` の `_test_` 付き「MOVED TO interpolation1d_tests.rs」）。

`REPOSITORY_RULES.md` の crate の説明から `interpolation2d.rs` と
`working_buffer.rs` を外し、`WorkingBuffer` を名指ししていた unsafe の規則を
「再利用する作業用の記憶域」一般の規則に書き換えた。

## 部分 6 のレビュー（part6-review エージェント）

実害のある欠陥はゼロ。D8-1〜D8-11 の削除された事前検査はすべて core 側の
同じ条件・同じ status に置き換わっていることを、置き換え先のコードまで
追って確認したとの報告。軽微な指摘 3 件のうち 2 件を 8c9b1f9 で直した。

3 件目（`utils.rs` の Gauss の 2 関数が、不正な segments でも先に
`legendre(n)` を計算する）は**直していない**。status も出力も正しく、
無効な入力の経路で n 次のルールを 1 回作るだけの無駄。直すには core の
`rule.piecewise` の検査を capi 側に写すことになり、部分 6 の方針
（検査をコアに寄せる）と逆になる。0.11.0 以降に回す。

## Task 8

- Step 1（古い記述と panic の経路、4 本の grep）: 1〜3 は出力なし。4 は
  決めたこと 20 の不変条件だけ（`basis.rs` の `flush().unwrap()` 3、
  `dlr.rs` の `unreachable!()` 2、`sampling.rs` の `assert_eq!` 2、
  `types.rs` の `partial_cmp(..).unwrap()`、`utils.rs` の `assert!` 2 と
  `.expect`）。
- Step 3（警告）: ビルドの警告は `part2-base-warn.txt` と正規化して一致。
  差は BLAS の行のほか、こちらが**減った** 2 件だけ
  （`vecs_approx_equal` never used、`unused variable: i`。どちらも削除した
  コードのもの）。rustdoc の警告も `ea718e3-doc-warn.txt` に対して減るだけ。
  narrowing で新しく出た「public documentation links to private item」3 件
  （`sve/result.rs` 2、`sve/strategy.rs` 1）は、intra-doc link を素の
  コードスパンに直して消した。
- Step 5: 0.10.0 の破壊的変更の一覧を
  `docs/superpowers/handoff/part7-breaking-changes.md` に書いた。
- Step 4（フルゲート）: 下に記録する。

### Step 4 フルゲート（`$LOG/p7-*.log`）

FMT OK。build 0、test 0、doc 0、sysblas 0、sysblas doc 0、cxx 0、
fortran 0、python 0。

| 層 | 実測 | 部分 6 |
| --- | --- | --- |
| cargo test（合計） | 574 passed / 0 failed / 1 ignored | 591 / 0 / 5 |
| doctest | 19 | 19 |
| system-blas のコア | 399 passed / 1 ignored | 416 / 5 |
| system-blas の doctest | 19 | 19 |
| C++ | 41788 アサーション（2 test case）＋ 417 アサーション（25 test case）、ctest 12/12 | 同じ |
| Fortran | ctest に含まれる 12/12 | 12/12 |
| Python | 97 passed | 97 |

cargo test と system-blas の減りはどちらも 17 で、Task 7 で消したテストの
分。ignored の 5 → 1 も同じ（`gauss_tests.rs` の `_test_` 4 本）。C++・
Fortran・Python は 1 つも変わっていない。C の挙動が変わっていないことの
確認になる。

ヘッダ: 最初のゲートで `assets/sparse_ir_capi.h` が 1 行だけ遅れていた
（上の逸脱 6 の行）。同期して 0d898af でコミット。以後 `include/` と
`assets/` と cbindgen の出力は同一。

`check_version.py` は macOS の system python3（3.9.6）では `str | None` を
解せず exit 1。部分 6 と同じ既知の事象で、この作業とは関係ない。
`python3.12` では exit 0（既知の Julia 0.8.4 != 0.9.0 の警告のみ）。

**部分 7 の C API の status の変化: なし。** 計画の表のとおり。

### ビルド生成物の混入と、その取り除き

ゲートに `--no-clean` を足したあと `fortran/_build_gfortran/` が残るように
なり、`.gitignore` の `*/_build/` が接尾辞つきのディレクトリに当たらない
ため、220 ファイル・4.4 MB（`libsparse_ir_capi.dylib` 2.3 MB を含む）が
3 つのコミットに入っていた。未 push だったので、`git filter-branch
--index-filter` でその経路だけを 9 コミットから落とした。

- 取り除く前の状態は `backup/p7-pre-artifact-cleanup`（旧 4256da9）に残して
  ある。作業ツリーの内容は、このディレクトリの外では取り除く前と 1 バイトも
  違わないことを `git diff --name-only` で確かめた（差分 220 件はすべて
  `fortran/_build_gfortran/` の下）。ソースは変わっていないので、上のゲートの
  結果はそのまま通用する。
- SHA が変わった: 3d14b09→9f9e701、a21d915→53ca308、c8a55c7→89bc569、
  253823f→c8cd930、db9aa0d→d520337、9e40d27→8c9b1f9、4256da9→0d898af。
  199d792 と 46d5067 は変わらない。上の表は新しい SHA に直してある。
- 再発しないように `.gitignore` に `*/_build_*/` を足した（b76e8c6）。

## 部分 7 のレビュー（part7-review エージェント）

**実害のある欠陥はゼロ。** 6 つのカテゴリすべてを、台帳の主張を信じずに
コードで確かめたとの報告。

1. 数値: 浮動小数点の演算順序が変わる箇所なし。`from_discretized_matrix` と
   `from_discretized_matrices_centrosymmetric` を f1b652b の capi 版と 1 行ずつ
   突き合わせ、引数・順序とも同一を確認。`f64::to_f64` は恒等なので新しく
   挟んだ変換もビット同一。中心対称版だけは even を完結させてから odd に
   進む形に変わったが、各要素の演算列は不変で、差が出うるのは「even の poly 化と
   odd の SVD が同時に失敗したときどちらの status が返るか」だけ。その条件
   （segments < 2）は C 側と `require_gauss_point_count` が先に弾くので到達不能。
   テスト側の変更は `.s` → `.s()` の機械置換のみで、許容誤差と参照値は不変。
2. C API: ヘッダの差は許可された doc 1 行のみ。2 ファイルは同一。新設の core の
   検査は C 側の既存の検査に完全に先回りされるので status は変わらない。
3. db9aa0d の削除: 削除した全項目名をリポジトリ全体に grep して残存参照ゼロ。
   `basis.rs` が使う `sign_changes`/`find_extrema` が自由関数のほうであり、
   その自由関数が f1b652b から 1 バイトも変わっていないことも確認。
4. アクセサ置換: 取り違えゼロ。増えたクローンは `poles().to_vec()` 1 件のみで、
   元も `.clone()` だった。
5. `default_matsubara_sampling_points_from_uhat`: `_impl` も
   `fence_matsubara_sampling` も `K` を一度も参照しない。`LogisticKernel` で
   固定しても結果は変わらない。
6. `#[allow(dead_code)] // Used in tests` は正直（`rank`・`rank_with_rtol` とも
   同ファイルのテストから実際に呼ばれている）。

軽微な指摘 6 件のうち、文書の実害があった 2 件を直した。

- `part7-breaking-changes.md` に `sve::utils`（`pub mod` → `pub(crate)`）の
  削除と、`SVEResult::from_discretized_matrix` ほかの追加を書き足した。
- `docs/INPLACE_OPTIMIZATION_PLAN.md` の表が、消えた `working_buffer.rs` を
  指したままだったのを `fitters/common.rs` に直した（`copy_to_contiguous` は
  もう存在しないので行ごと削除）。Task 8 Step 1 の grep から漏れていた。

残る 4 件は台帳の記述の正確さに関するもので、上の逸脱 4・7・8 と Task 7 の
削除リストの書き方に反映済み。
