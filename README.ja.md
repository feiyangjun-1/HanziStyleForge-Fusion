[简体中文](README.md) | [한국어](README.ko.md) | [日本語](README.ja.md) | [English](README.en.md)

# HanziStyleForge Fusion

Windows 向けの実験的な漢字フォント再構築ツールです。`target.ttf` から書体スタイルを学習し、`ref.otf` から漢字構造を取得して、インストール可能な TTF フォントを生成します。

> 長時間の無人実行を想定し、チェックポイント再開、安全停止、自動再試行に対応しています。

## 主な機能

- `fonts/target.ttf` から全体および局所的な書体スタイルを学習します。
- `refs/ref.otf` のデフォルト字形が持つすべての漢字を再構築します。
- 中国大陸、台湾、香港、日本、韓国、伝承字形などの参照フォントを利用できます。
- 対象フォントのラテン文字、数字、記号、仮名、ハングル、主要な OpenType データを可能な限り保持します。
- 学習、生成、候補選択、QA、ベクトル化、フォント構築を自動化します。

## 処理の概要

```text
target.ttf：スタイル
        +
ref.otf：漢字構造と対象範囲
        ↓
Style Encoder → VQ → Diffusion → Refiner / Retrieval / IDS
        ↓
候補選択 → QA → 輪郭変換 → TTF
```

プログラムは地域字形の正誤を判断しません。最終的な漢字構造は `ref.otf` のデフォルト Unicode `cmap` 字形に従います。

## 動作環境

- Windows 11 64-bit
- CUDA 対応 NVIDIA GPU
- Python 3.10 以降
- 150 GB 以上の空き容量を推奨

入力フォント：

```text
fonts\target.ttf
refs\ref.otf
```

静的フォントを推奨します。`target.ttf` には TrueType `glyf` テーブルが必要です。`ref.otf` は静的 TrueType または静的 CFF OTF を使用できます。可変フォント、TTC、OTC は使用しないでください。

## クイックスタート

1. リポジトリをダウンロードまたはクローンします。
2. スタイル元フォントを `fonts\target.ttf` に配置します。
3. 構造参照フォントを `refs\ref.otf` に配置します。
4. 環境をインストールします。

   ```text
   install_cuda130.bat
   ```

5. プロジェクトを確認します。

   ```text
   verify_project.bat
   ```

6. 完全な処理を開始または再開します。

   ```text
   run_months_resilient.bat
   ```

状態確認：

```text
run_status.bat
```

安全停止：

```text
request_safe_stop.bat
```

再開前に停止マーカーを削除します。

```text
clear_safe_stop.bat
run_months_resilient.bat
```

## 出力

主な出力：

```text
build\target-HanziStyleForge-Fusion.ttf
build\target-HanziStyleForge-Fusion.ttf.report.json
work_hanzistyleforge_fusion_months\qa\index.html
```

学習データ、チェックポイント、生成進捗は次の場所に保存されます。

```text
work_hanzistyleforge_fusion_months\
```

学習中はこのフォルダーを削除しないでください。

## 使用前の注意

- 完全な処理には数日、数週間、またはそれ以上かかる場合があります。
- リポジトリにはフォント、事前学習済み重み、第三者フォントデータセットは含まれません。
- 生成フォントには `target.ttf` と `ref.otf` の両方のライセンスが適用される場合があります。
- 学習、変更、再配布の権利を持つフォントだけを使用してください。
- 本プロジェクトは実験的です。公開前に QA ページと最終フォントを確認してください。

## 研究・参考資料

HanziStyleForge Fusion は独立実装です。以下のプロジェクトと論文はアーキテクチャ設計の参考です。上流のソースコード、事前学習済み重み、フォントデータセットは本リポジトリに同梱されていません。

| 出典 | 参考にした方向 |
|---|---|
| [zi2zi](https://github.com/kaonashi-tyc/zi2zi) | 漢字スタイル変換、内容とスタイルの分離 |
| [zi2zi-JiT](https://github.com/kaonashi-tyc/zi2zi-JiT) | 複数参照スタイル条件、拡散 Transformer |
| [FontDiffuser](https://github.com/yeungchenwa/FontDiffuser) | 拡散生成、マルチスケール内容集約、明示的スタイル制約 |
| [HanziGen](https://github.com/wangwenho/HanziGen) | VQ 表現と条件付き潜在拡散 |
| [VQ-Font](https://github.com/Yaomingshuai/VQ-Font) | 離散フォント token と構造認識強化 |
| [LF-Font / MX-Font](https://github.com/clovaai/fewshot-font-generation) | 局所部品スタイル、因子分解、複数専門家 |
| [DeepVecFont-v2](https://github.com/yizhiwang96/deepvecfont-v2) | Transformer ベクトル系列と輪郭補正 |
| [Efficient and Scalable Chinese Vector Font Generation via Component Composition](https://arxiv.org/abs/2404.06779) | 部品領域変換と大規模合成 |
| [cjkvi/cjkvi-ids](https://github.com/cjkvi/cjkvi-ids) | Unicode IDS 部品構造と局所領域ヒント |

引用は手法上の参考を示すだけであり、上流のコード、重み、データ、フォントをコピーする許可ではありません。第三者資料を使用する前に、現在のライセンスと利用条件を確認してください。

## ライセンス

プロジェクトコードのライセンスは [LICENSE](LICENSE) を参照してください。ユーザー提供フォント、生成フォント、第三者資料にはそれぞれの条件が適用されます。

## コントリビューション

Issue と Pull Request を歓迎します。第三者のコード、データ、モデルを追加する場合は、出典とライセンス情報を明記してください。
