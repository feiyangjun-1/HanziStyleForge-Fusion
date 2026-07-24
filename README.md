[简体中文](README.md) | [한국어](README.ko.md) | [日本語](README.ja.md) | [English](README.en.md)

# HanziStyleForge Fusion

一个面向 Windows 的实验性汉字字体重建工具：从 `target.ttf` 学习字体风格，从 `ref.otf` 获取汉字结构，并生成可安装的 TTF 字体。

> 项目适合长时间自动运行，支持检查点恢复、安全暂停和失败重试。

## 它能做什么

- 从 `fonts/target.ttf` 学习整体与局部字体风格。
- 按 `refs/ref.otf` 的默认字形重建其覆盖的全部汉字。
- 参考字体可以是大陆、台湾、香港、日本、韩国或其他字形标准。
- 尽量保留目标字体中的拉丁字母、数字、符号、假名、谚文及主要 OpenType 数据。
- 自动完成训练、生成、候选筛选、QA、矢量化和字体构建。

## 工作方式

```text
target.ttf：提供风格
        +
ref.otf：提供汉字结构和覆盖范围
        ↓
Style Encoder → VQ → Diffusion → Refiner / Retrieval / IDS
        ↓
候选筛选 → QA → 轮廓转换 → TTF
```

程序不会自行判断哪一种地区字形“更正确”。最终汉字结构以 `ref.otf` 的默认 Unicode `cmap` 字形为准。

## 环境要求

- Windows 11 64 位
- 支持 CUDA 的 NVIDIA GPU
- Python 3.10 或更高版本
- 建议至少 150 GB 可用磁盘空间

输入字体：

```text
fonts\target.ttf
refs\ref.otf
```

建议使用静态字体。`target.ttf` 应包含 TrueType `glyf` 表；`ref.otf` 可以是静态 TrueType 或静态 CFF OTF。不要使用可变字体、TTC 或 OTC。

## 快速开始

1. 下载或克隆本仓库。
2. 将目标字体放到 `fonts\target.ttf`。
3. 将参考字体放到 `refs\ref.otf`。
4. 双击安装环境：

   ```text
   install_cuda130.bat
   ```

5. 检查项目：

   ```text
   verify_project.bat
   ```

6. 开始或继续完整流程：

   ```text
   run_months_resilient.bat
   ```

查看状态：

```text
run_status.bat
```

安全暂停：

```text
request_safe_stop.bat
```

继续运行前清除暂停标记：

```text
clear_safe_stop.bat
run_months_resilient.bat
```

## 输出文件

主要输出：

```text
build\target-HanziStyleForge-Fusion.ttf
build\target-HanziStyleForge-Fusion.ttf.report.json
work_hanzistyleforge_fusion_months\qa\index.html
```

中间训练数据、检查点和生成进度保存在：

```text
work_hanzistyleforge_fusion_months\
```

不要在训练过程中删除该目录。

## 使用前须知

- 完整流程可能持续数天、数周或更久。
- 项目不包含字体文件、预训练权重或第三方字体数据集。
- 生成字体可能同时受 `target.ttf` 和 `ref.otf` 的许可证约束。
- 请仅使用你有权训练、修改和发布的字体。
- 本项目是实验性工具，正式发布字体前请检查 QA 页面并进行人工测试。

## 研究与参考来源

HanziStyleForge Fusion 是独立实现。以下项目和论文为架构设计提供了参考；本仓库不直接打包它们的源码、预训练权重或字体数据集。

| 来源 | 参考方向 |
|---|---|
| [zi2zi](https://github.com/kaonashi-tyc/zi2zi) | 汉字风格迁移、内容与风格分离 |
| [zi2zi-JiT](https://github.com/kaonashi-tyc/zi2zi-JiT) | 多参考风格条件、扩散 Transformer |
| [FontDiffuser](https://github.com/yeungchenwa/FontDiffuser) | 扩散生成、多尺度内容聚合、显式风格约束 |
| [HanziGen](https://github.com/wangwenho/HanziGen) | VQ 表示与条件潜空间扩散 |
| [VQ-Font](https://github.com/Yaomingshuai/VQ-Font) | 离散字体 token 与结构感知增强 |
| [LF-Font / MX-Font](https://github.com/clovaai/fewshot-font-generation) | 局部部件风格、因子分解、多专家 |
| [DeepVecFont-v2](https://github.com/yizhiwang96/deepvecfont-v2) | Transformer 矢量序列与轮廓修正 |
| [Efficient and Scalable Chinese Vector Font Generation via Component Composition](https://arxiv.org/abs/2404.06779) | 部件区域变换与大规模组合 |
| [cjkvi/cjkvi-ids](https://github.com/cjkvi/cjkvi-ids) | Unicode IDS 部件结构与局部区域提示 |

引用只表示方法层面的参考，不代表获得复制上游代码、权重、数据或字体的许可。使用任何第三方材料前，请检查其当前许可证与使用条款。

## 许可证

项目代码许可证见 [LICENSE](LICENSE)。用户提供的字体、生成字体和第三方材料仍受各自许可证约束。

## 贡献

欢迎提交 Issue 和 Pull Request。请在提交第三方代码、数据或模型时同时说明来源与许可证。
