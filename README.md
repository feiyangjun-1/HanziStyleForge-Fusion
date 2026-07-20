# HanziStyleForge Fusion
## 注意：本项目使用 GPL 3.0 协议。并且禁止将本项目及其生成的字体用于商业用途！

一个面向 Windows、支持长期断点恢复的汉字字形重建系统：**仅从 `target.ttf` 学习字体风格**，**仅从 `ref.otf` 获取汉字结构和目标字符范围**，重新生成参考字体覆盖的全部 Han 码位，并验证目标字体中的非 Han 字形和主要 OpenType 工程数据未被破坏。为避免协议冲突，请确保 target 字体与 ref 字体为同一协议。

## 核心数据职责

```text
fonts/target.ttf  -> 只用于学习风格
refs/ref.otf      -> 只用于提供 Han 结构和目标字符范围
```

训练采用 `target.ttf` 自重建：

```text
target 真实字形 -> target 去风格结构代理 -> 模型 -> target 真实字形真值
```

推理阶段才读取 `ref.otf`：

```text
ref Han 结构 -> target 风格模型 -> 重建后的 target 风格 Han 字形
```

系统不需要人工同形字清单、CN/非 CN 分类或跨字体配对监督。`hanzistyleforge/contract.py` 会阻止参考字体路径进入训练，也会阻止目标字体被当作生成结构来源。

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

## Windows 11 快速开始

1. 解压到本地短路径，例如：

   ```text
   C:\FontWork\HanziStyleForge-Fusion
   ```

2. 放入字体：

   ```text
   fonts\target.ttf
   refs\ref.otf
   ```

3. 安装隔离 CUDA 环境：

   ```text
   install_cuda130.bat
   ```

4. 检查环境、字体、覆盖范围、配置和数据流契约：

   ```text
   verify_project.bat
   ```

5. 开始或恢复完整长期流程：

   ```text
   run_months_resilient.bat
   ```

6. 只读查看状态：

   ```text
   run_status.bat
   ```

7. 请求在下一个持久检查点安全停止：

   ```text
   request_safe_stop.bat
   ```

   再次运行 `run_months_resilient.bat` 即可恢复；脚本会自动清除已经完成的停止请求。

8. QA 报告生成后可运行：

   ```text
   open_qa.bat
   ```

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
| [cjk-decomp](https://github.com/amake/cjk-decomp) | 局部残差区域的可选拆分提示 | 多许可证；本项目对内置数据选择 Apache-2.0 |

详细说明见 [METHOD_REFERENCES.md](METHOD_REFERENCES.md)，第三方再分发声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

参考公开方法不等于获得复制实现、数据集、字体或模型权重的许可。若后续加入任何上游材料，必须保留其版权声明并遵守当前许可证。

## 温馨提示

本项目完全使用 GPT5.6-Sol 生成，作者完全不懂代码。欢迎任何人 pull request 对本项目改进。所有 pull request 也会由 GPT 进行审查。
