# Rust tutorial 2a 実行台帳

計画: `docs/superpowers/plans/2026-09-27-sparse-ir-rust-tutorial-2a.md`
設計書: `docs/superpowers/specs/2026-09-26-sparse-ir-rust-tutorial-design.md`
起点: `refactor/typed-errors-core`（部分 7 の最後、`ee75610`）
ブランチ: `docs/rust-tutorial`（ローカルのみ。**push していない**）
作業ツリー: `/Users/hiroshi/projects/sparse-ir/worktrees/wt-te`
ログ: `/Users/hiroshi/projects/sparse-ir/worktrees/logs`

## コミット

| SHA | 内容 | Task |
| --- | --- | --- |
| 06b7276 | Lay out the Rust tutorial: book, runnable examples, plotting | 1, 2 |
| b573781 | Add the sparse sampling tutorial and make the book's code blocks run | 3, 8 |
| 4092d93 | Add the transformation tutorial | 4 |
| 33ea0ed | Add the DLR tutorial | 5 |
| 7fef3d7 | Add the sparse modeling tutorial | 6 |
| 74452d1 | Add the analytic continuation tutorial | 7 |
| d3cee39 | Add CI for the tutorial and Pages for the book | 9 |

本体（`sparse-ir/`、`sparse-ir-capi/`）のコードは 1 行も変えていない。本体の
`Cargo.toml` に `exclude` を 2 つ足しただけで、`Cargo.lock` は変わっていない。

## 計画からの逸脱と、その理由

1. **Task 6 の前提が違った。** 計画 §1.1 は spm を「Python 版と同じ FISTA」と
   書いていたが、`spm_py.ipynb` が実際に使うのは `admmsolver` による ADMM
   （`ConstrainedLeastSquares` + `L1Regularizer(10**-1.8, size)` +
   `NonNegativePenalty`）で、非負制約と総和則を**硬い制約**として課す。
   `admmsolver` は Rust に移す対象ではないので、Rust 版は制約を落とし、IR
   係数への L1 だけを FISTA でかける同じ問題を解くことにした。落とした二つ
   の制約は本文の "What this port leaves out" に書いてある。照合用の Python
   も同じ FISTA を独立に書き、同じ入力を読ませてある（`reference_spm.py`
   の冒頭に理由を書いた）。
2. **Task 7 の前提も違った。** 計画は「1 次元求根」と書いているが、
   `analytic_continuation_py.ipynb` に求根は**一つもない**。実際の中身は
   特異値の減衰、切断 SVD、Ridge、ρₗ の非コンパクト性、Lorentz 実軸基底の
   五つで、そのとおりに移した。
3. **`roots.rs`（`bisect`・`brent`）を削除した。** 2 の結果、2a のどの例も
   使わなくなったため。ユーザーの判断は「削除する」。
4. **ω 空間ではなく IR 係数に L1 をかけた（Task 6）。** ω 空間の非負 LASSO は
   反復を増やすほど解がデルタの櫛に崩れ（3000 反復で L2 誤差 0.055、100000
   反復で 0.65）、早期打ち切りが実質の正則化になっていた。IR 係数への L1 は
   収束先が滑らかで疎な解になる。
5. **FISTA の収束判定を反復数固定にした（Task 6）。** 相対変化は 1e-8 付近で
   振動して止まるので、閾値では収束しない。20000 反復固定・`tol = 0` にして、
   最後の相対変化が 1e-6 未満であることを assert する形にした。
6. **乱数を CSV にコミットした（Task 6, 7）。** ノートブックは
   `numpy.random` の Mersenne Twister を使う。Rust で再現する方法がなく、
   別の乱数にすると Rust と Python の結果が数値として比べられなくなるので、
   引きを一度だけ取って `input/<example>/*.csv` に固定した。CI が実行時に何も
   ダウンロードしない、という制約にも合う。
7. **半円の重なり積分を一般化した（Task 7）。** `semicircle_overlaps` は
   `shifted_semicircle_overlaps(basis, center, half_width, weight)` の薄い
   包みになった。絶縁体模型が半円二つの和で書けるようになる。
   `sparse_sampling_demo` の出力が bit 一致することを確かめてある。
8. **`verification-summary.json` の書き方（Task 9)。** 計画は「書くところまで」
   としか決めていなかった。比較 1 件ごとに追記して毎回書き直す形にした。
   失敗した比較そのものがファイルに残るのは、この形だけ。

## 数値の照合

Python 参照値は公開版（`sparse-ir==2.1.4`、`pylibsparseir==0.9.1`、
`numpy==2.5.3`）。許容誤差は `tests/verification.rs` に、実測値をコメントで
添えて書いてある。157 件の比較すべてが通る。

| 例 | 最悪の相対ずれ | どこ |
| --- | --- | --- |
| transformation | 2.9e-15 | 往復の誤差 |
| sparse_sampling_demo | 3.0e-14 | Matsubara 標本の Im G |
| dlr | 3.8e-14 | DLR から作った G(iν) の実部 |
| spm | 6.3e-14 | λ 走査の `min_rho` |
| analytic_continuation | 3.4e-10 | 切断なしの ρ(ω)（1/s_last が 4e7 倍する） |

この表は `data/verification-summary.json` から出したもの（157 件、失敗 0）。

## 検証

- `docs/tutorial-code/scripts/check.sh --run`: fmt、clippy（`-D warnings`）、
  単体テスト、5 本の実行、157 件の照合、`mdbook test` 全 8 章 — すべて通る。
### 本体のフルゲート（`$LOG/gate2a-*.log`）

FMT OK。build 0、test 0、doc 0、sysblas 0、sysblas doc 0、header sync OK、
cxx 0、fortran 0、python 0。

| 層 | 実測 | 部分 7 |
| --- | --- | --- |
| cargo test（合計） | 574 passed / 0 failed / 1 ignored | 同じ |
| doctest | 19 | 19 |
| system-blas のコア | 399 passed / 1 ignored | 同じ |
| system-blas の doctest | 19 | 19 |
| C++ | 41788 アサーション（2 test case）＋ 417 アサーション（25 test case）、ctest 12/12 | 同じ |
| Fortran | ctest に含まれる 12/12 | 12/12 |
| Python | 97 passed | 97 |

部分 7 のゲートと 1 つも違わない。本体の `Cargo.toml` の `exclude` を触った
影響がないことの確認になる（計画 Task 10 の宿題）。

`check_version.py` が exit=1 なのは部分 7 と同じ理由で、macOS の system
python3（3.9.6）が `str | None` を解釈できないため。`uv run --python 3.12
python check_version.py` は通る（Julia の版が 0.8.4 のままという既知の警告
だけ）。このブランチとは無関係。

## 残っていること

- 2b（応用 8 本。Yuki Nagai さん作のものは除く）。`tutorial-applied` の job は
  すでにあるが中身は空。
- PR は 2a と 2b を合わせて 1 本。**まだ push していない。**
