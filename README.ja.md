# HanziStyleForge Fusion 2.2

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

Windows を中心に設計された、長時間実行とチェックポイント再開に対応する漢字グリフ再構築システムです。**`target.ttf` からは書体スタイルだけを学習し**、**`ref.otf` からは漢字の構造と対象文字範囲だけを取得**します。参照フォントが収録するすべての Han コードポイントを再生成し、ターゲットフォントの非 Han グリフと主要な OpenType データが保持されていることを検証します。

> 現在の状態は研究・エンジニアリング向け Alpha です。生成フォントを配布する前に、目視確認、組版アプリでのテスト、ライセンス確認を行ってください。

## コアデータ契約

```text
fonts/target.ttf  -> スタイル学習専用
refs/ref.otf      -> Han 構造と対象文字範囲専用
```

学習は `target.ttf` の自己再構成だけで行います。

```text
target の実グリフ -> target のスタイル除去構造代理 -> モデル -> target の実グリフ正解
```

生成時にのみ `ref.otf` を読み込みます。

```text
ref の Han 構造 -> target スタイルモデル -> target スタイルの再構築 Han グリフ
```

手作業の同形字リスト、中国規格/非中国規格の分類、クロスフォントのペア教師は不要です。`hanzistyleforge/contract.py` は、参照フォントのパスが学習に混入した場合や、ターゲットフォントが生成構造として使用された場合に処理を停止します。

## 主な機能

- `ref.otf` の既定 Unicode `cmap` に含まれるすべての Han コードポイントを再構築します。
- 参照フォントは、中国大陸、台湾、香港、日本、韓国、伝承字形、その他の字形規格を使用できます。必要な形が参照フォントの既定グリフであることが条件です。
- 複数の実 target グリフから、全体スタイルと局所スタイルを学習します。
- VQ グリフコードブック、潜在拡散、決定論的安全ベースライン、実グリフ局所検索、部品残差、高解像度 Refiner、トポロジーゲート、輪郭補正を組み合わせます。
- 拡散出力が target 正解ではなくスタイル除去構造代理へ近づく場合、自動停止します。
- 学習、生成、グリフ単位の補正、QA、フォント構築を永続チェックポイントから再開できます。
- 最終フォントは `target.ttf` を基礎にし、再構築 Han グリフだけを追加・再マップします。
- 出力前に非 Han `cmap`、glyph ID、輪郭、メトリクス、UVS、レイアウト表、ヒンティング関連表を検証します。

## 推奨環境

```text
Windows 11 64-bit
12 GB 以上の VRAM を持つ NVIDIA GPU
Python 3.10-3.14 64-bit
ローカル SSD
150 GB 以上の空き容量
```

入力条件：

- `fonts/target.ttf`: `glyf` 表を持つ静的 TrueType。可変フォントは不可。
- `refs/ref.otf`: 静的 TrueType TTF/OTF または静的 CFF OTF。
- TTC/OTC、および実行時の `locl` に依存しないと目的地域の字形が現れない参照フォントは避けてください。
- 配布パッケージにフォントや学習済み重みは含まれません。

## Windows 11 クイックスタート

1. 短いローカルパスへ展開します。

   ```text
   C:\FontWork\HanziStyleForge-Fusion
   ```

2. フォントを配置します。

   ```text
   fonts\target.ttf
   refs\ref.otf
   ```

3. CUDA 環境をインストールします。

   ```text
   install_cuda130.bat
   ```

4. 環境、フォント、対象範囲、設定、データフロー契約を検証します。

   ```text
   verify_project.bat
   ```

5. 完全な長時間ワークフローを開始または再開します。

   ```text
   run_months_resilient.bat
   ```

6. 状態を読み取り専用で表示します。

   ```text
   run_status.bat
   ```

7. 次の永続チェックポイントで安全停止を要求します。

   ```text
   request_safe_stop.bat
   ```

   再開時は `run_months_resilient.bat` をもう一度実行してください。完了済みの停止要求は自動的に解除されます。

8. QA レポートが生成された後は次を実行できます。

   ```text
   open_qa.bat
   ```

## 残されている Windows ランチャー

| ファイル | 用途 |
|---|---|
| `install_cuda130.bat` | `.venv` を作成し、依存関係と CUDA を検証 |
| `verify_project.bat` | セルフテストとプロジェクト検証 |
| `run_months_resilient.bat` | 完全な再開可能ワークフローを開始・再開 |
| `request_safe_stop.bat` | 次の永続チェックポイントで停止 |
| `run_status.bat` | 状態を読み取り専用表示 |
| `open_qa.bat` | HTML QA レポートを開く |

高度な段階は Python CLI から利用できます。

```powershell
.venv\Scripts\python.exe hanzistyleforge.py --help
.venv\Scripts\python.exe hanzistyleforge.py --config config_fusion_months_12gb.json fusion-train
.venv\Scripts\python.exe hanzistyleforge.py --config config_fusion_months_12gb.json fusion-generate
```

## 本番設定

主設定ファイルは `config_fusion_months_12gb.json` です。

```json
{
  "training": {
    "workers": 4
  },
  "fusion": {
    "style_encoder": {
      "batch_size": 8
    }
  }
}
```

Style Encoder は品質ゲート付き早期終了に対応します。最低 100 epoch 学習した後、最近の正負スタイル類似度が健全な状態で、24 epoch にわたり有意な検証改善がなければ自動終了します。互換性のあるチェックポイントと `history.csv` は再利用されます。

現在の段階のチェックポイントを破棄する意図がない限り、途中で画像サイズ、モデルチャネル、スタイル次元、潜在チャネル、コードブックサイズを変更しないでください。

## ワークフロー

```text
入力と CUDA の確認
-> target/ref グリフのレンダリング
-> データフロー契約の検証
-> target 局所スタイルアトラスの構築
-> Style Encoder 学習
-> target VQ コードブック学習
-> 決定論的安全ベースライン学習
-> 多解像度潜在拡散学習
-> 実 target の難例マイニング
-> 高解像度 Refiner 学習
-> 輪郭 Transformer 学習
-> ref が収録する全 Han の生成
-> トポロジーとスタイルによる候補選択
-> 再開可能なグリフ単位補正
-> QA
-> SDF/TrueType ベクトル化
-> 最終フォントの構築と検証
```

処理は数週間から数か月かかる場合があります。主要段階と生成済みグリフはチェックポイント化されます。回復可能なエラーは resilient ランチャーが再試行しますが、永続的な品質保護エラーは自動再試行しません。

## 対象範囲と非 Han 保護

既定設定：

```json
{
  "scope": {
    "mode": "reference_han",
    "include_compatibility_ideographs": true
  }
}
```

参照フォントのすべての Han コードポイントには、生成結果または安全フォールバックが必要です。`require_complete=true` の場合、欠落が 1 文字でもあれば正式出力を停止します。

元の target glyph ID を守るため新規 glyph を追加するので、次の制限があります。

```text
target glyph 数 + 追加 Han glyph 数 < 65,536
```

最終構築では、ラテン、キリル、かな、ハングル、数字、句読点、記号、非 Han 輪郭、メトリクス、Unicode 変異シーケンス、GSUB/GPOS/GDEF/BASE/kern、TrueType ヒンティング関連表を検証します。

## 出力

```text
build\target-HanziStyleForge-Fusion.ttf
build\target-HanziStyleForge-Fusion.ttf.report.json
work_hanzistyleforge_fusion_months\generated\coverage.json
work_hanzistyleforge_fusion_months\generated\selection.csv
work_hanzistyleforge_fusion_months\refined\selection.csv
work_hanzistyleforge_fusion_months\qa\index.html
```

## 研究・参照コードの出典

HanziStyleForge Fusion は独立実装です。以下の公開プロジェクトと論文は設計方針の参考です。本リポジトリはそれらのソースコード、学習済み重み、フォントデータセットを**同梱・再配布しません**。

| 上流プロジェクト | 参考にした方向 | 上流ライセンス/状態 |
|---|---|---|
| [zi2zi](https://github.com/kaonashi-tyc/zi2zi) | 漢字スタイル変換、内容とスタイルの条件分離 | Apache-2.0 |
| [zi2zi-JiT](https://github.com/kaonashi-tyc/zi2zi-JiT) | 複数参照条件と拡散 Transformer | MIT ソフトウェアライセンスに加え、上流のフォント成果物付加条項あり。最新条項を確認してください |
| [FontDiffuser](https://github.com/yeungchenwa/FontDiffuser) | ノイズ除去拡散、多尺度内容集約、明示的スタイル制約 | 本リリース準備時、上流リポジトリにライセンスファイルを確認できませんでした。許可なくコードや重みをコピーしないでください |
| [HanziGen](https://github.com/wangwenho/HanziGen) | VQ-VAE と潜在拡散によるフォント補完 | Apache-2.0 |
| [VQ-Font](https://github.com/Yaomingshuai/VQ-Font) / [論文](https://doi.org/10.1609/aaai.v38i15.29577) | 離散フォント token 事前分布と構造認識強化 | コードや重みをコピーする前に最新ライセンスを確認してください |
| [LF-Font / MX-Font 統合リポジトリ](https://github.com/clovaai/fewshot-font-generation) | 局所部品スタイル、因子分解、複数エキスパート | MIT。上流の一部モジュールには別の出典表示があります |
| [DeepVecFont-v2](https://github.com/yizhiwang96/deepvecfont-v2) | Transformer ベースのベクトル列と輪郭補正 | コードは MIT。上流フォントデータセットには別の非商用制限があります |
| [Efficient and Scalable Chinese Vector Font Generation via Component Composition](https://arxiv.org/abs/2404.06779) | 部品領域変換と大規模合成 | 論文参照。関連コード・データの条件は別途確認してください |
| [cjk-decomp](https://github.com/amake/cjk-decomp) | 局所残差領域のための任意分解ヒント | 複数ライセンス。本配布物の同梱データは Apache-2.0 オプションを使用 |

詳細は [METHOD_REFERENCES.md](METHOD_REFERENCES.md)、第三者再配布表示は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。

公開手法を参考にすることは、実装、データセット、フォント、モデル重みをコピーする許可ではありません。上流素材を追加する場合は、著作権表示を保持し、最新ライセンスに従ってください。

## ライセンスとフォント権利

```text
Copyright 2026 feiyangjun_
```

HanziStyleForge のソースコードとプロジェクト文書は、個別表示された第三者素材を除き、[Apache License 2.0](LICENSE) で提供されます。

このライセンスはユーザー提供フォントの権利を付与しません。`target.ttf` と `ref.otf` のライセンスが、学習、改変、派生フォント作成、配布を許可しているか確認してください。生成フォントとチェックポイントは、片方または両方の入力フォントライセンスの対象になる場合があります。

## 関連文書

- [アーキテクチャ](ARCHITECTURE.md)
- [データフロー契約](DATA_FLOW.md)
- [手法参照](METHOD_REFERENCES.md)
- [第三者通知](THIRD_PARTY_NOTICES.md)
- [テスト報告](TEST_REPORT.md)
- [コントリビューション](CONTRIBUTING.md)
- [セキュリティポリシー](SECURITY.md)
- [変更履歴](CHANGELOG.md)
