<div align="center">

# XEmbodied

**面向大规模具身环境的几何与物理线索增强基础模型**

[![arXiv](https://img.shields.io/badge/arXiv-2604.18484-B31B1B.svg)](https://arxiv.org/abs/2604.18484)
[![官方代码](https://img.shields.io/badge/Official%20Code-xiaomi--research-blue)](https://github.com/xiaomi-research/xiaomi_spatialembodied)
[![模型权重](https://img.shields.io/badge/Model-ModelScope-624aff)](https://www.modelscope.cn/models/xzhong/xembodied-v0)

**[English](README.md)**

</div>

## 概览

XEmbodied（发布名 **Xiaomi-SpatialEmbodied**）是面向自动驾驶与机器人数据闭环的云端具身 VQA 基础模型。模型基于 **Qwen3-VL-30B-A3B-Instruct**，增加两条互补的增强路径：

- **3DA（3D Adapter）**：用 VGGT 提取几何特征，通过 cross-attention residual 注入 Qwen 的视觉 token，保留原序列长度并补充深度、布局和相对位置信息。
- **EIEA（Efficient Image-Embodied Adapter）**：把 occupancy、3D box、map 等工具输出蒸馏成紧凑的 TOR embedding，在不暴露原始工具轨迹的情况下提供物理先验。
- **渐进式领域课程**：从通用多模态能力逐步过渡到空间/3D 和具身数据，降低灾难性遗忘。

论文在 18 个空间、语义、具身与 affordance 基准上报告了一致提升。

## 官方资源（以此为准）

**官方仓库是架构、环境安装、模型权重、推理方式和完整用法的权威来源：**

| 资源 | 链接 |
|---|---|
| **官方仓库 / README** | **https://github.com/xiaomi-research/xiaomi_spatialembodied** |
| 官方 README（英文） | https://github.com/xiaomi-research/xiaomi_spatialembodied/blob/main/README.md |
| 官方 README（中文） | https://github.com/xiaomi-research/xiaomi_spatialembodied/blob/main/README_CN.md |
| 论文 | https://arxiv.org/abs/2604.18484 |
| 模型权重 | https://www.modelscope.cn/models/xzhong/xembodied-v0 |

> 环境配置、checkpoint、推理命令、完整 benchmark 表格与最新更新请直接参考官方 README。本仓库只保留简要概览。

## 本仓库内容

本仓库是论文相关的实验/配套代码，主要包括：

| 目录 | 内容 |
|---|---|
| `data/` | DriveBench、DriveLMM-o1、Ego3D-Bench、OmniDrive、MapLM-v2、VLAD-Bench、RoboAfford、PartAfford、RoboRefit 等数据转换与统计 |
| `evaluation/` | 各 benchmark 评测脚本与统一 system prompt |
| `training/` | EIEA 数据管线与训练入口（含 3DA / box / BEV 变体） |
| `model/` | 训练/评测使用的模型侧工具 |
| `tools/` | 成本基准、paired bootstrap / McNemar 统计等实验工具 |
| `docs/` | 架构、训练、归一化、数据来源与 benchmark 覆盖说明 |
| `scripts/infer_demo.py` | 小型 demo 入口 |

模型实现与规范用法请以官方仓库为准。

## 引用

```bibtex
@misc{qian2026xembodiedfoundationmodelenhanced,
  title={XEmbodied: A Foundation Model with Enhanced Geometric and Physical Cues for Large-Scale Embodied Environments},
  author={Kangan Qian and ChuChu Xie and Yang Zhong and Jingrui Pang and Siwen Jiao and Sicong Jiang and Zilin Huang and Yunlong Wang and Kun Jiang and Mengmeng Yang and Hao Ye and Guanghao Zhang and Hangjun Ye and Guang Chen and Long Chen and Diange Yang},
  year={2026},
  eprint={2604.18484},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2604.18484}
}
```

## 作者

Kangan Qian\*、ChuChu Xie\*、Yang Zhong<sup>✉,\*</sup>、Jingrui Pang、Siwen Jiao、Sicong Jiang、Zilin Huang、Yunlong Wang、Kun Jiang†、Mengmeng Yang、Hao Ye<sup>✉,†</sup>、Guanghao Zhang、Hangjun Ye、Guang Chen、Long Chen、Diange Yang†

小米公司汽车与机器人部 · 清华大学 · 新加坡国立大学 · 麦吉尔大学 · 威斯康星大学麦迪逊分校

✉ 项目负责人 · \* 同等贡献 · † 通讯作者
