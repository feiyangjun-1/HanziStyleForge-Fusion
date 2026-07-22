# HanziStyleForge Fusion

一个面向 Windows、支持长期断点恢复的汉字字形重建系统：**仅从 `target.ttf` 学习字体风格**，**仅从 `ref.otf` 获取汉字结构和目标字符范围**，重新生成参考字体覆盖的全部 Han 码位，并验证目标字体中的非 Han 字形和主要 OpenType 工程数据未被破坏。

目前 Github 上几个类似的项目使用起来比较麻烦，对代码小白相当不友好，问过 ChatGPT 也看不懂如何运行，所以我让 GPT5.6-Sol Pro 帮我生成了这个项目。

这个项目完全从头开始训练字体风格，经历多重工序，理论上可以做到开箱即用，运行脚本（虽然要花很长时间）即可直接生成符合对应地区字形、无需人工调整的 ttf 字体文件。

## 主要功能

- 重建 `ref.otf` 默认 Unicode `cmap` 中的全部 Han 码位。
- 参考字体可以采用大陆、台湾、香港、日本、韩国、传承字形或其他字形标准，前提是所需形态就是参考字体的默认字形。
- 从多个真实 target 字形中学习全局和局部风格。
- 融合 VQ 字形码本、潜空间扩散、确定性安全基线、真实局部字形检索、部件残差、高分辨率 Refiner、拓扑门控和轮廓修正。
- 当扩散预测逐渐靠近无风格结构代理而不是 target 真值时，自动停止并写出错误报告。
- 训练、生成、逐字修正、QA 和字体构建都支持持久检查点恢复。
- 最终字体从 `target.ttf` 开始构建，只追加并重映射重建后的 Han 字形。
- 输出前复核非 Han `cmap`、glyph ID、轮廓、度量、UVS、布局表和提示相关表。

## 环境要求

正式运行建议：

```text
Windows 11 64 位
12GB 或更高显存的 NVIDIA GPU
Python 3.10-3.14 64 位
本地 SSD
至少 150GB 可用空间
```

输入要求：

- `fonts/target.ttf`：静态 TrueType 字体，包含 `glyf` 表，不是可变字体。
- `refs/ref.otf`：静态 TrueType TTF/OTF 或静态 CFF OTF。
- 不建议使用 TTC/OTC，也不要使用必须依赖运行时 `locl` 才能显示目标地区字形的参考字体。
- 项目包不包含字体文件或预训练权重。
- 为避免协议冲突，请确保 target 字体与 ref 字体为同一协议。

## 使用方法

1. 下载[本仓库源码](https://github.com/feiyangjun-1/HanziStyleForge-Fusion/archive/refs/heads/main.zip)解压到本地短路径，例如：

   ```text
   C:\FontWork\HanziStyleForge-Fusion
   ```

2. 放入需要学习风格的 target 字体和字形参考 ref 字体，必须将字体重命名为 target 和 ref：

   ```text
   fonts\target.ttf
   refs\ref.otf
   ```
   请尽量使用黑体作为 ref 字体。推荐使用静态版思源黑体 Regular，不要使用通过自动程序改造过的字体。

4. 安装隔离 CUDA 环境：

   ```text
   install_cuda130.bat
   ```

5. 检查环境：

   ```text
   verify_project.bat
   ```
   出现 successful 即可。

6. 开始运行：

   ```text
   run_months_resilient.bat
   ```

7. 查看目前状态（可选）：

   ```text
   run_status.bat
   ```

8. 安全暂停并保存进程：

   ```text
   request_safe_stop.bat
   ```

9. 继续保存的进程
    
   ```text
   run_months_resilient.bat
   ```

  脚本会自动清除已经完成的停止请求。

## 正式配置

主要配置文件：`config_fusion_months_12gb.json`。

当前 Style Encoder 默认值：

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

Style Encoder 支持质量门控早停：至少训练 100 个 epoch；之后若连续 24 个 epoch 没有显著验证改善，并且最近的正负风格相似度指标保持健康，就会自动结束。兼容的检查点和 `history.csv` 会直接复用。

除非准备丢弃当前阶段检查点，否则不要在阶段中途修改图像尺寸、模型通道数、风格维度、潜通道数或码本大小。

## 完整流程

```text
检查输入和 CUDA
-> 渲染 target/ref 字形
-> 验证数据流契约
-> 建立 target 局部风格图谱
-> 训练 Style Encoder
-> 训练 target VQ 码本
-> 训练确定性安全基线
-> 训练多分辨率潜空间扩散
-> 挖掘真实 target 困难样本
-> 训练高分辨率 Refiner
-> 训练轮廓 Transformer
-> 生成 ref 覆盖的全部 Han
-> 拓扑与风格候选选择
-> 可恢复的逐字修正
-> QA
-> SDF/TrueType 矢量化
-> 构建并验证最终字体
```

整个流程可能运行数周或数月。主要阶段和每个已生成字形都会保存检查点。可恢复错误由 resilient 脚本自动重试；持久质量保护错误不会自动重试。

## 覆盖范围和非 Han 保护

默认范围：

```json
{
  "scope": {
    "mode": "reference_han",
    "include_compatibility_ideographs": true
  }
}
```

每个参考 Han 码位都必须获得生成结果或安全回退。启用 `require_complete=true` 时，任何缺失都会阻止正式输出。

由于构建器通过追加新 glyph 来保护 target 原 glyph ID，因此必须满足：

```text
target glyph 数 + 新增 Han glyph 数 < 65,536
```

最终构建会复核 target 的拉丁、俄文、假名、谚文、数字、标点、符号、非 Han 轮廓、度量、Unicode 变体序列、GSUB/GPOS/GDEF/BASE/kern 和 TrueType 提示相关表。

## 输出文件

```text
build\target-HanziStyleForge-Fusion.ttf
build\target-HanziStyleForge-Fusion.ttf.report.json
work_hanzistyleforge_fusion_months\generated\coverage.json
work_hanzistyleforge_fusion_months\generated\selection.csv
work_hanzistyleforge_fusion_months\refined\selection.csv
work_hanzistyleforge_fusion_months\qa\index.html
```

## 研究与参考代码来源

HanziStyleForge Fusion 是独立实现。下列公开项目和论文为架构方向提供了参考；本仓库**未打包或再分发**它们的源码、预训练权重或字体数据集。

| 上游项目 | 参考方向 | 上游授权/状态 |
|---|---|---|
| [zi2zi](https://github.com/kaonashi-tyc/zi2zi) | 汉字风格迁移、内容与风格条件分离 | Apache-2.0 |
| [zi2zi-JiT](https://github.com/kaonashi-tyc/zi2zi-JiT) | 多参考风格条件和扩散 Transformer 方向 | MIT 软件许可，另有上游字体产物附加条款；使用前应查看当前版本 |
| [FontDiffuser](https://github.com/yeungchenwa/FontDiffuser) | 去噪扩散、多尺度内容聚合和显式风格约束 | 本版本整理时未在上游仓库看到许可证文件；未经许可不要复制源码或权重 |
| [HanziGen](https://github.com/wangwenho/HanziGen) | VQ-VAE 与潜空间扩散的字体补全流程 | Apache-2.0 |
| [VQ-Font](https://github.com/Yaomingshuai/VQ-Font) / [论文](https://doi.org/10.1609/aaai.v38i15.29577) | 离散字体 token 先验和结构感知增强 | 复制源码或权重前检查当前仓库许可证 |
| [LF-Font / MX-Font 统一仓库](https://github.com/clovaai/fewshot-font-generation) | 局部部件风格、因子分解和多专家 | MIT；部分上游模块有独立来源声明 |
| [DeepVecFont-v2](https://github.com/yizhiwang96/deepvecfont-v2) | 基于 Transformer 的矢量序列和轮廓修正 | 代码 MIT；上游字体数据集另有非商业限制 |
| [Efficient and Scalable Chinese Vector Font Generation via Component Composition](https://arxiv.org/abs/2404.06779) | 部件区域变换和大规模组合 | 论文参考；相关代码和数据需分别检查条款 |
| [cjkvi/cjkvi-ids](https://github.com/cjkvi/cjkvi-ids) | 为局部残差区域提供标准 Unicode IDS 部件布局 | `ids.txt` 由程序直接从上游下载，不随项目分发；上游说明其遵循适用的 CHISE 条款 |

详细说明见 [METHOD_REFERENCES.md](METHOD_REFERENCES.md)，第三方再分发声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 温馨提示

本项目完全使用 GPT5.6-Sol 生成，作者完全不懂代码。欢迎任何人 pull request 对本项目改进。所有 pull request 也会由 GPT 进行审查。
