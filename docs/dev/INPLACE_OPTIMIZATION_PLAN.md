# In-place Evaluate/Fit Optimization Plan

## 概要

C-API (`sparse-ir-capi`) の `eval` / `fit` 系関数のパフォーマンス最適化計画。
主な目標は、内部でのメモリ確保とコピーを削減し、FFI境界でのオーバーヘッドを最小化すること。

## 背景

現在の実装では以下のオーバーヘッドが存在:

1. **C-API境界でのコピー**: C側のポインタからRust `Tensor` への読み込み、結果のコピー
2. **内部での次元操作**: `movedim`, `reshape().to_tensor()` 等で新しいメモリを確保
3. **複素数の分離/結合**: 実部・虚部の抽出と結合時に新しいバッファを確保

## 実装フェーズ

### Phase 1: fitter.rs - 2D in-place メソッド ✅ 完了

**追加されたメソッド:**

| メソッド | 説明 |
|---------|------|
| `evaluate_2d_to(&self, backend, coeffs_2d, out)` | 実数係数 → 実数値 (in-place) |
| `fit_2d_to(&self, backend, values_2d, out)` | 実数値 → 実数係数 (in-place) |
| `evaluate_complex_2d_to(&self, backend, coeffs_2d, out)` | 複素数係数 → 複素数値 (in-place) |
| `fit_complex_2d_to(&self, backend, values_2d, out)` | 複素数値 → 複素数係数 (in-place) |

**特徴:**
- 出力テンソル `out` に直接書き込み
- 形状の事前検証
- 既存メソッドと同一の結果を保証（テスト済み）

### Phase 2: sampling.rs - N-D in-place メソッド ✅ 完了

**追加されたメソッド (TauSampling):**

| メソッド | 説明 |
|---------|------|
| `evaluate_nd_to<T>(&self, backend, coeffs, dim, out)` | N次元係数 → N次元値 (in-place) |
| `fit_nd_to<T>(&self, backend, values, dim, out)` | N次元値 → N次元係数 (in-place) |

**現状の制限:**
- 内部での次元置換 (`movedim`) によるコピーは残存
- 最終結果のコピーのみ削減

### Phase 3: matsubara_sampling.rs ✅ 完了

**追加されたメソッド (MatsubaraSampling):**

| メソッド | 説明 |
|---------|------|
| `evaluate_nd_to<T>(&self, backend, coeffs, dim, out)` | T係数 → Complex値 (in-place) |
| `fit_nd_to(&self, backend, values, dim, out)` | Complex値 → Complex係数 (in-place) |

**追加されたメソッド (MatsubaraSamplingPositiveOnly):**

| メソッド | 説明 |
|---------|------|
| `evaluate_nd_to(&self, backend, coeffs, dim, out)` | f64係数 → Complex値 (in-place) |
| `fit_nd_to(&self, backend, values, dim, out)` | Complex値 → f64係数 (in-place) |

### Phase 4: C-API統合 🔄 基盤完了

**完了した基盤:**

| コンポーネント | 追加内容 |
|--------------|---------|
| `gemm.rs` | `matmul_par_to_viewmut()` - DViewMutへの直接書き込み |
| `fitter.rs` | `evaluate_2d_to_viewmut()` - DViewMutを受け取る2D評価 |
| `sparse-ir-capi/utils.rs` | `create_viewmut_2d_row_major()` - 生ポインタからDViewMut作成 |

**残りの作業（後回し）:**
- C-APIのeval/fit関数を完全に統合
- N-D版のDViewMut対応
- ベンチマークによる効果測定

**使用例（将来）:**
```rust
// After (in-place)
let coeffs_view = DView::new_unchecked(coeffs_ptr, mapping);
let mut out_view = DViewMut::new_unchecked(out_ptr, out_mapping);
sampling.evaluate_nd_to(&coeffs_view, dim, &mut out_view);
```

### Phase 5: 内部最適化 ✅ 完了（シンプル版）

**目標:**
- `movedim` でのコピーを削減
- strided viewを使った次元置換

**実装アプローチ: シンプル版**

複雑なバッファ再利用（SamplingContext）は削除し、シンプルな実装を採用:

| コンポーネント | 追加内容 |
|--------------|---------|
| `fitters/common.rs` | `copy_from_contiguous()` - 連続 → strided コピー |
| `sampling.rs` | `evaluate_nd_inplace()` - uninit Vecを使用した効率的な実装 |

**追加されたメソッド (TauSampling):**

| メソッド | 説明 |
|---------|------|
| `evaluate_nd_inplace(backend, coeffs, dim, out)` | 一時バッファを都度allocate（0クリアなし） |

**方針:**
- 一時バッファは `Vec::with_capacity` + `set_len` でuninit allocate
- 即座に書き込むので0クリア不要
- シンプルで保守しやすい

## テスト結果

```
sparse-ir::fitter::tests - 20件のテスト全通過
sparse-ir::sampling::tests - 4件の新規テスト全通過
sparse-ir::matsubara_sampling::tests - 4件の新規テスト全通過
全体: 249件中249件通過（5件ignored）
```

## ブランチ情報

- **ブランチ名**: `feature/inplace-eval-fit-methods`
- **ベース**: `main`

## 次のステップ

1. [x] Phase 3: matsubara_sampling.rsへのin-placeメソッド追加
2. [ ] Phase 4: C-APIでのDViewMut活用
3. [ ] ベンチマーク実施と効果測定
4. [ ] PRの作成とレビュー

## 参考リンク

- [mdarray DViewMut documentation](https://docs.rs/mdarray/)
- [faer strided matrix support](https://github.com/sarah-ek/faer-rs)
