# Git Bisect ガイド - SVEメモリ問題の調査

## 問題の概要

`beta=1e+5, epsilon=1e-10`のパラメータでSVE計算が約250GBのメモリを使用する問題が発生しています。

## 二分探索の手順

### 1. 現在の状態を確認

```bash
# 現在のコミットで問題が発生することを確認（BAD）
./capi_benchmark/bisect_manual.sh HEAD
# 結果: BAD (memory issue exists)
```

### 2. Git Bisectを開始

```bash
cd /Users/hiroshi/projects/sparse-ir/sparseir-rust
git bisect start
git bisect bad HEAD  # 現在のコミットは問題あり
git bisect good <古いコミットハッシュ>  # 問題が発生しないコミット
```

### 3. 各コミットでテスト

git bisectが自動的に中間のコミットをチェックアウトするので、各コミットで：

```bash
./capi_benchmark/bisect_manual.sh HEAD
```

結果に応じて：
- 問題あり（メモリ > 100GB）: `git bisect bad`
- 問題なし（メモリ < 100GB）: `git bisect good`
- テストできない（ビルドエラーなど）: `git bisect skip`

### 4. 問題が導入されたコミットを特定

git bisectが問題が導入されたコミットを見つけたら、そのコミットの変更内容を確認：

```bash
git show <コミットハッシュ>
git bisect reset  # bisectを終了
```

## 手動テストスクリプト

`bisect_manual.sh`を使用して特定のコミットをテストできます：

```bash
./capi_benchmark/bisect_manual.sh <commit-hash>
```

戻り値:
- 0: 問題あり（BAD）
- 1: 問題なし（GOOD）
- 125: テストできない（SKIP）

## 注意事項

- 各コミットでライブラリをビルドする必要があるため、時間がかかります
- 古いコミットではC APIが存在しない可能性があります
- ビルドエラーが発生した場合は`git bisect skip`を使用
