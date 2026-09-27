# Rust tutorial 2b 実行台帳

計画: `docs/superpowers/plans/2026-09-27-sparse-ir-rust-tutorial-2b.md`
設計書: `docs/superpowers/specs/2026-09-26-sparse-ir-rust-tutorial-design.md`
起点: `docs/rust-tutorial`（2a の最後、`d0e0397`）
ブランチ: `docs/rust-tutorial`（ローカルのみ。**push していない**）
作業ツリー: `/Users/hiroshi/projects/sparse-ir/worktrees/wt-te`
ログ: `/Users/hiroshi/projects/sparse-ir/worktrees/logs`

## コミット

| SHA | 内容 | Task |
| --- | --- | --- |
| b2396ca | Pin the conventions the applied tutorials will stand on | 0 |
| 91515b4 | Add the second-order perturbation tutorial | 1 |
| 7fd6146 | Add the GW tutorial | 2 |
| 6f6ba27 | docs(tutorial): add the Liechtenstein exchange-interaction example | 3 |
| 0858494 | docs(tutorial): add the orbital magnetic susceptibility example | 4 |
| fa59727 | docs(tutorial): add the DMFT/IPT applied example | 5 |
| e1ee1b3 | docs(tutorial): add the two-particle self-consistency applied example | 6 |
| 3bf09ee | docs(tutorial): add the fluctuation-exchange applied example | 7 |
| ff26645 | docs(tutorial): add the Eliashberg applied example | 8 |
| aaa65b7 | ci(tutorial): run the applied examples | 9 |

`d0e0397..HEAD` で変えたのは 164 ファイル、21873 行追加・11 行削除。すべて
`docs/` と `.github/workflows/tutorial.yml` の中で、`sparse-ir/`・
`sparse-ir-capi/` のコードも本体の `Cargo.toml`／`Cargo.lock` も 1 行も
触っていない。

## 計画からの逸脱と、その理由

1. **応用は 8 本。** Yuki Nagai さん作のノートブック（`sparse_ir_nagai*`）は
   ユーザーの判断で除いた。残る 8 本が second_order_perturbation、gw、
   liechtenstein、orbital_magnetic_susceptibility、dmft_ipt、tpsc、flex、
   eliashberg_holstein。重い 3 本（dmft_ipt、tpsc／flex、eliashberg）は
   単発と走査に分け、走査は `SPARSEIR_TUTORIAL_SCANS=1` で明示的に有効化した
   ときだけ走る。
2. **Task 0 を足した。** 計画は例から始まっていたが、8 本すべてが同じ二つの
   規約（τ 格子が Python の `[0, β)` ではなく `[−β/2, β/2]`、統計が違っても
   τ 格子が一致する）に乗る。先に `IrMesh`・`reverse_tau`・`evaluate_rows`
   を固定して単体テストを付け、それから例を書いた。`reverse_tau` は Python の
   `g_tau[::-1]`、すなわち `G(β − τ) = ζ G(−τ)` を、符号付きの置換として
   実装したもの。
3. **`roots.rs` を戻した（Task 6）。** 2a の Task 7 で「どの例も使わない」と
   判断して消したが、tpsc と flex は SciPy の `brentq` と同じ意味の 1 次元
   求根で `U_sp`／`U_ch` と化学ポテンシャルを決める。`xtol` も反復上限も
   SciPy の既定値に合わせてある。
4. **`Lattice` から `Bases` を切り出した（Task 8）。** Eliashberg は格子を
   持たない（Holstein 模型の局所版）が、`FiniteTempBasisSet` に相当する
   「1 つの SVE から両統計の基底と標本格子」は要る。`Lattice` は `Bases` を
   包むだけになり、既存のアクセサは委譲で残した。出力は bit 一致する。
5. **τ の折り返しを明示した（Task 8）。** Python の `g_tau[g_tau > 0] = 0` を
   そのまま `[−β/2, β/2]` の格子に当てると、負 τ 側の正の値まで潰れる。
   `clamp_to_negative` は比較の前に値を `[0, β)` 側へ折り返す
   （`τ > 0` ならそのまま、そうでなければ符号反転）。ここを間違えると
   Π の `G(τ)G(β − τ)` が死に、反復は常伝導解に収束して gap が 1e-10 に
   落ちる（実際に一度そうなった）。本文にも警告として書いてある。
   比較は NumPy の複素の辞書式順序（実部、次に虚部）をそのまま再現した。
6. **乱数を CSV にコミットした（Task 8）。** Eliashberg の初期 Σ のノイズは
   `input/eliashberg_holstein/noise.csv` に固定してある。理由は 2a と同じで、
   Mersenne Twister を Rust で再現できないことと、CI が実行時に何も
   ダウンロードしない、という制約。
7. **dmft_ipt の許容誤差だけ緩い。** 自己無撞着ループが丸めを増幅するため、
   `g_im` は 5e-6（実測 3.3e-7）、`z` は 1e-6（実測 1.1e-7）。半充填・対称
   状態密度なので `g_re` は粒子・正孔対称性で消えるはずで、残るのは同じ
   増幅された丸め（実測 4.9e-7）。`assert_negligible` で `g_im` に対する
   相対量として見ている。ほかの 7 本はすべて 1e-9 以下。
8. **CI は応用を pull request では走らせない（Task 9）。** `tutorial-applied`
   の job は main への push、毎週月曜 03:17 UTC の schedule、手動実行の
   ときだけ走る（`timeout-minutes: 120`）。`tutorial-basics` と同じ
   `shared-key: tutorial` のキャッシュを使う。schedule で落ちたときだけ
   `tutorial-applied` ラベルの issue を 1 本立てる（すでに開いていれば
   コメントを足す）――停止 1 回につき issue 1 本で、週 1 本にはしない。

## 数値の照合

Python 参照値は 2a と同じ公開版（`sparse-ir==2.1.4`、`pylibsparseir==0.9.1`、
`numpy==2.5.3`）。許容誤差は `tests/verification.rs` に実測値のコメント付きで
書いてある。`data/verification-summary.json` は **520 件、失敗 0**
（2a の 157 件を含む）。

| 例 | 比較数 | 最悪の相対ずれ | どこ |
| --- | --- | --- | --- |
| transformation | 31 | 2.9e-15 | error |
| sparse_sampling_demo | 22 | 3.0e-14 | g_iv_im |
| dlr | 26 | 3.8e-14 | g_iv_dlr_re |
| spm | 22 | 6.3e-14 | min_rho |
| analytic_continuation | 56 | 3.4e-10 | semielliptic_full |
| second_order_perturbation | 33 | 3.6e-10 | cond_tau |
| gw | 59 | 2.2e-14 | difference |
| liechtenstein | 20 | 4.5e-13 | j0 |
| orbital_magnetic_susceptibility | 23 | 5.8e-14 | chi_re |
| dmft_ipt | 26 | 4.9e-07 | g_re（上の 7 を見よ） |
| dmft_ipt_scan | 17 | 1.4e-14 | sigma_im_u34 |
| tpsc | 34 | 4.0e-14 | chi_spin |
| tpsc_scan | 22 | 1.2e-13 | u_ch |
| flex | 41 | 5.0e-12 | residual |
| flex_scan | 27 | 4.4e-10 | residual |
| eliashberg_holstein | 37 | 7.6e-12 | tau |
| eliashberg_holstein_scan | 24 | 1.7e-09 | c_over_t |

反復回数は数値ではなく整数として一致を要求している。Eliashberg は単発が
171 反復、走査は 20 回の求解すべてで反復回数が Python と同じ
（193, 75, 174, 93, 229, 123, 337, 182, 630, 348, 4476, 3374, 1555, 13, 84,
12, 25, 12, 22, 12）。

## 検証

- `docs/tutorial-code/scripts/check.sh --run --scans --release`: fmt、
  clippy（`-D warnings`）、単体テスト、13 本のバイナリ＋走査の実行、
  520 件の照合（`tests/verification.rs` の 17 テスト）、`mdbook test` 全
  15 章 — すべて通る（exit 0）。

### 本体のフルゲート（`$LOG/gate2b-20260927-165710.log`）

FMT OK。build 0、test 0、doc 0、sysblas 0、sysblas doc 0、header sync OK、
cxx 0、fortran 0、python 0。

| 層 | 実測 | 2a |
| --- | --- | --- |
| cargo test（合計） | 574 passed / 0 failed / 1 ignored | 同じ |
| doctest | 19 | 19 |
| system-blas のコア | 399 passed | 399 |
| system-blas の doctest | 19 | 19 |
| C++ | 41788 アサーション（2 test case）＋ 417 アサーション（25 test case）、ctest 2/2 | 同じ |
| Fortran | 12/12 | 12/12 |
| Python | 97 passed | 97 |

2a のゲートと 1 つも違わない。2b が `docs/` と
`.github/workflows/tutorial.yml` しか触っていないことの確認になる
（計画 Task 10 の宿題）。

## 残っていること

- PR は 2a と 2b を合わせて 1 本。**まだ push していない。**
- 本文の図は `docs/plotting` で作り、`docs/book/src/tutorials/*.png` として
  コミット済み。CI は図を作り直さない（作図は matplotlib と uv に依存する）。
