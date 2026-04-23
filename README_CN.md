<div align="center">
<h1><b>XEmbodied</b></h1>
</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2604.18484-B31B1B.svg?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.18484)

</div>

<div align="center">

**[<a href="README.md">English</a>]**
**[<a href="https://www.kaggle.com/models/zhongyangtony/xembodied/">模型权重</a>]**

</div>

> **Authors:**
>
> Kangan Qian<sup>✉,\*</sup>, ChuChu Xie<sup>\*</sup>, Yang Zhong<sup>\*</sup>, Jingrui Pang, Siwen Jiao, Sicong Jiang, Zilin Huang, Yunlong Wang, Kun Jiang<sup>†</sup>, Mengmeng Yang, Hao Ye<sup>†</sup>, Guanghao Zhang, Hangjun Ye, Guang Chen, Long Chen, Diange Yang<sup>†</sup>
>
> **Affiliations:**
>
> Tsinghua University &nbsp; Automotive and Robotics, Xiaomi Corporation &nbsp; National University of Singapore &nbsp; McGill University &nbsp; University of Wisconsin–Madison
>
> ✉ Project Leader &nbsp; \* Equal contribution &nbsp; † Corresponding author

**XEmbodied: A Foundation Model with Enhanced Geometric and Physical Cues for Large-Scale Embodied Environments** ([arXiv 2604.18484](https://arxiv.org/abs/2604.18484)) 的官方开源实现。

我们提出了 XEmbodied，一个面向自动驾驶与机器人 VQA 数据闭环系统的云端基础模型。XEmbodied 通过联合增强领域语义与三维理解能力，同时保持通用能力，实现了大规模场景挖掘与标注的鲁棒性。核心设计包括 **3D Adapter (3DA)**——一种几何连接器，通过专用 3D 令牌将三维基础模型的表征注入多模态大语言模型；不同于简单附加深度或点云的辅助输入方式，3DA 显式对齐三维结构与语言推理，赋予模型内生的三维感知能力。我们进一步提出 **Efficient Image-Embodied Adapter (EIEA)**，将原始工具输出蒸馏为紧凑的令牌摘要并无缝回注到 MLLM 上下文中，规避了 LLM 的可解释性负担。

**核心贡献：**
- 提出 XEmbodied，一个融合内禀几何表征与物理线索交互的云端具身闭环 VQA 通用模型
- 提出 3DA 与 EIEA 驱动的隐式物理线索对齐，实现自适应几何先验注入与高效物理增强推理
- 设计渐进式领域课程以实现鲁棒适应并减少遗忘，在 18 个基准上验证了一致的具身理解增益

## News
- 2026/04/27: XEmbodied 项目代码正式开源
- 2026/04/20: XEmbodied 论文上线 arXiv 预印本平台

## 实验结果

### 表 1：空间与 3D 理解
表中最佳结果加**粗**。

| 模型 | Ego3DBench ACC | Ego3DBench RMSE | SURDS | VLADBench | STRIDE-QA |
|---|---|---|---|---|---|
| **闭源模型** | | | | | |
| GPT-4o [67] | 52.70 | 19.20 | 13.31 | 56.00 | 14.70 |
| Gemini-1.5 [82] | 53.50 | 19.62 | 32.77 | 54.23 | - |
| **开源模型** | | | | | |
| Qwen2.5-VL-7B [5] | 41.10 | 30.36 | 12.61 | 50.75 | 1.41 |
| Qwen2.5-VL-32B [5] | 57.30 | 15.87 | 38.82 | 61.33 | 6.58 |
| Qwen3-VL-A3B-30B [4] | 53.15 | 13.84 | 38.69 | 59.81 | 8.69 |
| Qwen3-VL-32B [4] | **59.93** | 15.70 | 41.53 | 61.64 | 6.00 |
| **空间模型** | | | | | |
| UniVG-R1-7B [6] | 46.82 | 53.85 | 1.96 | 46.51 | 4.61 |
| PR1-OCR-2B [101] | 38.88 | 11.60 | 13.81 | 44.65 | 8.18 |
| PR1-Detection-3B [101] | 36.74 | 62.61 | 33.84 | 56.08 | 4.35 |
| PR1-Counting-2B [101] | 40.32 | 51.95 | 13.69 | 48.52 | 1.94 |
| PR1-Grounding-2B [101] | 39.54 | 12.34 | 13.96 | 43.67 | 7.15 |
| **具身模型** | | | | | |
| DriveMM [36] | 49.73 | 12.68 | 8.73 | 47.45 | 1.12 |
| Cosmos-R1 [2] | 45.62 | 23.41 | 10.05 | 55.93 | 1.03 |
| Mimo-Embodied [29] | 53.62 | 16.57 | 24.04 | 55.82 | 7.90 |
| **本模型** | | | | | |
| XEmbodied (Best) | 55.28 | **9.25** | **83.83** | **68.61** | **27.76** |

### 表 2：语义与推理
表中最佳结果加**粗**。

| 模型 | DriveBench | DriveLMM-o1 | MapLM-v2 | LingoQA | Omnidrive |
|---|---|---|---|---|---|
| **闭源模型** | | | | | |
| GPT-4o [67] | 47.97 | 57.84 | 57.81 | 53.58 | 4.54 |
| Gemini-1.5 [82] | - | - | 58.44 | 61.30 | 2.73 |
| **开源模型** | | | | | |
| Qwen2.5-VL-7B [5] | 43.29 | 37.81 | 55.10 | 53.20 | 2.53 |
| Qwen2.5-VL-32B [5] | 53.06 | 55.04 | 57.47 | 43.70 | 0.00 |
| Qwen3-VL-A3B-30B [4] | 46.22 | 57.17 | 53.63 | 47.00 | 0.31 |
| Qwen3-VL-32B [4] | 51.70 | 55.29 | 46.77 | 54.30 | 0.00 |
| **空间模型** | | | | | |
| UniVG-R1-7B [6] | 48.75 | 51.42 | 39.13 | 39.20 | 1.20 |
| PR1-OCR-2B [101] | 50.73 | 47.68 | 44.05 | 48.50 | 4.07 |
| PR1-Detection-3B [101] | 41.35 | 44.81 | 62.47 | 56.08 | 0.10 |
| PR1-Counting-2B [101] | 39.72 | 50.94 | 47.43 | 48.52 | 2.00 |
| PR1-Grounding-2B [101] | 52.03 | 53.83 | 51.00 | 50.30 | 4.85 |
| **具身模型** | | | | | |
| DriveMM [36] | 44.50 | 65.91 | 52.18 | 37.70 | 1.10 |
| Cosmos-R1 [2] | 35.80 | 54.66 | 55.33 | 52.70 | 1.23 |
| Mimo-Embodied [29] | 52.95 | 40.31 | 65.23 | **68.60** | 4.90 |
| **本模型** | | | | | |
| XEmbodied (Best) | **53.18** | **77.01** | **78.55** | 65.70 | **25.43** |

### 表 3：具身与可供性
表中最佳结果加**粗**。

| 模型 | Affordance-2K | Robo-Afford | Cosmos-R1 | Embodied-R1 | RoboRefitBench | VABench-Point | Where2place |
|---|---|---|---|---|---|---|---|
| **闭源模型** | | | | | | | |
| GPT-4o [67] | 2.70 | 3.80 | 67.40 | 0.35 | 13.40 | 0.40 | 0.44 |
| Gemini-1.5 [82] | 5.19 | 4.20 | 71.10 | - | 35.50 | 0.80 | 0.89 |
| **开源模型** | | | | | | | |
| Qwen2.5-VL-7B [5] | 8.70 | 3.00 | 72.50 | 0.25 | 76.40 | 0.75 | 0.90 |
| Qwen2.5-VL-32B [5] | 9.10 | 3.25 | 72.50 | 0.00 | 77.60 | 1.45 | 1.25 |
| Qwen3-VL-A3B-30B [4] | 8.85 | 3.65 | 74.50 | 0.05 | **83.95** | 1.10 | 1.51 |
| Qwen3-VL-32B [4] | 6.90 | 4.00 | 75.50 | 0.05 | 83.15 | 1.84 | 1.50 |
| **具身模型** | | | | | | | |
| Cosmos-R1 [2] | 6.63 | 1.95 | 72.00 | 0.05 | 34.75 | 0.35 | 0.45 |
| Mimo-Embodied [29] | 8.85 | 3.25 | **81.00** | 0.00 | 78.35 | 1.85 | 1.25 |
| **本模型** | | | | | | | |
| XEmbodied (Best) | **78.50** | **4.35** | 76.00 | **3.80** | 87.15 | **3.50** | **2.30** |

## TODO

- [x] 发布模型代码
- [x] 发布模型权重
- [x] 发布推理代码
- [ ] 发布数据集下载
- [ ] 发布评测代码
- [ ] 发布训练代码

## 微调模型

| 模型 | 基座模型 | 权重 |
|------|----------|------|
| XEmbodied | Qwen3-VL-30B-A3B-Instruct | [Kaggle](https://www.kaggle.com/models/zhongyangtony/xembodied/) |

## 数据集

XEmbodied 在多种具身与通用 VQA 数据集上训练：

| 类别 | 数据集 |
|------|--------|
| 驾驶 VQA | LingoQA, SURDS, MapLM, DriveVQA, DriveLMM-o1, Omnidrive, BDD100K |
| 通用 VQA | RefCOCO, Flickr, VQAv2, GQA|
| 机器人操作 | RoboRefit, Cosmos-R1, RoboVQA |
| 3D / 空间 | Visual Trace, VSI-590K, SPAR-7M, OpenSpaces |
| 具身可交互性 | Part Affordance |

## 项目结构

```
BaseModel-open/
├── xembodied/
│   ├── 3DA/
│   │   └── my_qwen3_vggt_xattnv2/         # Qwen3-VL-MoE-VGGT（3DA 几何连接器）
│   │       ├── modeling_qwen3_vl_moe_vggt.py  # 核心几何增强模型定义
│   │       ├── configuration_qwen3_vl_moe.py
│   │       ├── modular_qwen3_vl_moe.py
│   │       ├── plugin/
│   │       │   ├── qwen3_vl_moe_vggt_register.py  # MS-Swift 自定义模型注册
│   │       │   └── save_qwen3_vl_moe_vggt.py      # 合并 VGGT 权重并生成融合检查点
│   │       └── vggt/                        # VGGT 骨干网络代码
│   ├── EIEA/
│   │   ├── infer_with_EIEA_3D.py            # EIEA 物理线索注入推理流水线
│   │   ├── models/
│   │   │   ├── qwen3_vl_vggt_register.py    # 集成 TOR 物理嵌入注入的模型注册文件
│   │   │   ├── qwen3_vl_moe_vggt.py         # 支持物理-几何双增强的模型定义
│   │   │   └── ...
│   │   ├── tor_embeds_cache_fb.pt           # 预缓存物理先验 TOR 嵌入
│   │   └── tor_embeds_cache.pt
│   ├── infer_demo.py                        # 基础 VLM 推理入口
│   └── infer_demo_show.py                   # 可视化推理演示脚本
└── tools/                                   # 数据处理与评测工具集
    ├── convert_*.py                         # 各数据集转换脚本
    ├── infer_debug/
    │   ├── infer_demo_show.py               # 基于 MS-Swift 引擎的推理脚本
    │   ├── option_parser.py                 # 选项解析工具
    │   ├── filter_think_format.py           # R1 类型模型的 think 标签过滤
    │   └── box_match.py                     # 框格式正则匹配
    ├── eval_tools/                          # 评测结果聚合脚本
    ├── vis/                                 # 结果可视化
    ├── data_statistics/                     # 数据统计与过滤
    ├── build_model/                         # 模型构建/保存/调试脚本
    └── reward_debug/                        # 奖励模型调试
```

## 模型架构

XEmbodied 为统一的自动驾驶与具身智能基础模型框架，几何连接器（3DA）与物理线索注入范式（EIEA）深度协同，整体基于 Qwen3-VL-30B-A3B-Instruct 构建：

### 1. 3DA：几何空间增强连接器

3D Adapter (3DA) 是一种几何连接器，通过专用 3D 令牌将三维基础模型（VGGT）的表征注入多模态大语言模型。不同于简单附加深度或点云的辅助输入方式，3DA 显式对齐三维结构与语言推理，赋予模型内生的三维感知能力，使其无需依赖外部深度模型即可原生理解具身场景的三维几何结构。

核心组件：
- **VGGTSpatialEncoder**：加载 VGGT 聚合器提取图像多层空间特征，精简冗余头结构以降低显存消耗
- **VGGTEmbeddingMerger**：完成 3D 空间令牌与原生 2D 视觉令牌的对齐融合，通过双线性插值、令牌合并、RMSNorm + MLP 投影与 Cross-Attention 机制，将三维几何线索注入视觉语言模型

该模型通过 MS-Swift 注册，支持一站式微调与推理，是 XEmbodied 几何感知能力的基础载体。

### 2. EIEA：高效图像-具身适配器

已有工作利用具身专用工具（如占据网格、3D 框、高精地图线索）显式引导 VQA 推理，但此类显式工具调用存在效率低、工具输出与语言推理过程对齐差的问题。为解决这些局限，EIEA 将原始工具输出蒸馏为紧凑的令牌摘要（TOR 嵌入），并通过 `masked_scatter` 无缝回注到 MLLM 上下文中，规避了 LLM 的可解释性负担。

EIEA 与 3DA 共享模型底层结构，共同构成 XEmbodied 的完整具身感知能力。

## 环境配置

### 1. 基础环境

```bash
conda create -n xembodied python=3.10
conda activate xembodied

# 安装 PyTorch
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
```

### 2. MS-Swift（训练与推理框架）

```bash
pip install ms-swift
pip install qwen_vl_utils==0.0.14
pip install trl -U
pip install vllm -U
pip install transformers==4.49.0
```

### 3. EIEA 依赖

```bash
pip install peft==0.14.0
pip install aenum sentencepiece protobuf
```

## 快速开始：路径配置

运行推理脚本前需配置模型路径，仓库默认采用相对路径规划，仅需更新指定占位路径即可。

### 所需模型与权重说明
| 项目 | 默认位置 | 重要说明 |
|------|----------|----------|
| Qwen3-VL-30B-A3B-Instruct | `Qwen/Qwen3-VL-30B-A3B-Instruct` | 基座模型，支持自动下载 |
| 训练完成模型权重 | Kaggle Model Scope 下载 | 可推理的完整权重：https://www.kaggle.com/models/zhongyangtony/xembodied/ |
| VGGT 注册文件 | `xembodied/3DA/.../qwen3_vl_moe_vggt_register.py` | MS-Swift 模型注册文件 |
| EIEA 注册文件 | `xembodied/EIEA/models/qwen3_vl_vggt_register.py` | 集成物理嵌入注入的注册文件 |
| TOR 嵌入缓存 | `xembodied/EIEA/tor_embeds_cache_fb.pt` | EIEA 自带预缓存文件，无需额外下载模型 |

### 配置步骤
1. **下载基座 VLM**：下载 [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) 或由 MS-Swift 自动下载
2. **获取可用模型权重**：从 Kaggle Model Scope 下载训练完成的 XEmbodied 权重，[XEmbodied](https://www.kaggle.com/models/zhongyangtony/xembodied/)
3. **EIEA 物理嵌入准备**：仓库已内置预缓存 TOR 嵌入文件，无需额外下载预训练模型，直接使用即可
4. **更新脚本路径**：在推理脚本中修改模型路径为本地权重路径，保持目录结构可直接适配相对路径

## 推理

### 方式一：3DA 几何增强推理
使用 `xembodied/infer_demo.py`，基于 MS-Swift 调用 XEmbodied 几何连接器：

```bash
cd XEmbodied
python ./xembodied/infer_demo.py
```

脚本配置项：
```python
# 启用 XEmbodied 几何增强模型
USE_CUSTOM_MODEL = True

# 自定义模型配置（使用 Kaggle 下载的训练完成权重）
CUSTOM_MODEL_CFG = {
    "model_path": "Kaggle 下载的模型权重路径",
    "model_type": "qwen3_vl_moe_vggt",
    "register_file": "<注册文件路径>",
}

# 原生模型配置
NATIVE_MODEL_CFG = {
    "model_path": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "model_type": "qwen3_vl_moe",
}

# 推理参数
INFER_CFG = {
    "max_tokens": 512,
    "temperature": 0,
    "user_prompt": "你的提示词",
    "image_paths": ["<图片路径>"]
}
```

### 方式二：EIEA 物理-几何双增强推理
使用 `xembodied/EIEA/infer_with_EIEA_3D.py` 运行完整物理线索注入推理流水线：

```bash
cd xembodied/EIEA
python infer_with_EIEA_3D.py
```

该脚本实现物理与几何双线索协同推理，内置预缓存 TOR 嵌入，无需下载额外预训练模型，自动完成场景物理先验注入与 3D 几何感知推理。

## 工具集

| 目录/脚本 | 功能 |
|-----------|------|
| `convert_*.py` | 将 NuScenes、OmniDrive、LingoQA 等具身数据集转为训练 JSONL 格式 |
| `parse_*.py` | 解析评测原始输出为标准化格式 |
| `infer_debug/` | 推理调试与格式过滤工具 |
| `eval_tools/` | 具身场景评测结果聚合 |
| `vis/` | 交互与感知结果可视化 |
| `data_statistics/` | 数据统计、过滤与采样 |
| `build_model/` | 模型加载、保存与调试 |
| `reward_debug/` | GRPO 奖励模型调试 |

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{qian2026xembodiedfoundationmodelenhanced,
      title={XEmbodied: A Foundation Model with Enhanced Geometric and Physical Cues for Large-Scale Embodied Environments},
      author={Kangan Qian and ChuChu Xie and Yang Zhong and Jingrui Pang and Siwen Jiao and Sicong Jiang and Zilin Huang and Yunlong Wang and Kun Jiang and Mengmeng Yang and Hao Ye and Guanghao Zhang and Hangjun Ye and Guang Chen and Long Chen and Diange Yang},
      year={2026},
      eprint={2604.18484},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.18484},
}
```
