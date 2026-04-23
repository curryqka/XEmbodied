# XEmbodied

[English](README.md)

**XEmbodied: A Foundation Model with Enhanced Geometric and Physical Cues for Large-Scale Embodied Environments** ([arXiv 2604.18484](https://arxiv.org/abs/2604.18484)) 的官方开源实现。

本仓库开源 XEmbodied 具身智能基础模型的完整代码，面向大规模具身场景统一构建**几何空间感知**与**物理先验线索**增强能力：将 VGGT 3D 空间令牌与物理驱动的 TOR（Textual Object Rationale）嵌入协同融入 Qwen3-VL-MoE 架构，从几何结构与物理约束双维度提升视觉语言模型在具身环境中的空间认知、物理交互推理与场景理解能力。

## News
- 2026/04/27: XEmbodied 项目代码正式开源
- 2026/04/20: XEmbodied 论文上线 arXiv 预印本平台

## 项目结构

```
BaseModel-open/
├── xembodied/
│   ├── 3DA/
│   │   └── my_qwen3_vggt_xattnv2/         # Qwen3-VL-MoE-VGGT 模型（Cross-Attention 几何融合主干）
│   │       ├── modeling_qwen3_vl_moe_vggt.py  # 核心几何增强模型定义
│   │       ├── configuration_qwen3_vl_moe.py
│   │       ├── modular_qwen3_vl_moe.py
│   │       ├── plugin/
│   │       │   ├── qwen3_vl_moe_vggt_register.py  # MS-Swift 自定义模型注册
│   │       │   └── save_qwen3_vl_moe_vggt.py      # 合并 VGGT 权重并生成基础融合检查点
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
    ├── agent_tools/                         # Agent 工具调用数据生成
    ├── eval_tools/                          # 评测结果聚合脚本
    ├── vis/                                 # 结果可视化
    ├── data_statistics/                     # 数据统计与过滤
    ├── build_model/                         # 模型构建/保存/调试脚本
    └── reward_debug/                        # 奖励模型调试
```

## 模型架构

XEmbodied 为**统一的自动驾驶和具身智能基础模型框架**，几何增强主干与物理线索注入范式深度协同，整体基于 Qwen3-VL-30B-A3B-Instruct 构建：

### 1. Qwen3-VL-MoE-VGGT（几何空间增强主干）
作为 XEmbodied 的核心几何感知分支，通过 Cross-Attention 机制将 VGGT 3D 空间令牌融入 Qwen3-VL 架构，让模型原生具备具身场景的三维几何结构理解能力，无需依赖外部深度模型即可完成空间推理。

核心组件：
- **VGGTSpatialEncoder**：加载 VGGT 聚合器提取图像多层空间特征，精简冗余头结构以降低显存消耗
- **VGGTEmbeddingMerger**：完成 3D 空间令牌与原生 2D 视觉令牌的对齐融合，通过双线性插值、令牌合并、投影与交叉注意力机制，将三维几何线索注入视觉语言模型

该模型通过 MS-Swift 注册，支持一站式微调与推理，是 XEmbodied 几何感知能力的基础载体。

### 2. EIEA 物理线索注入推理范式
EIEA（External Implicit Embodied Agent）是 XEmbodied 面向具身场景设计的**物理先验注入机制**，核心是向模型注入场景物理约束、对象交互合理性等物理线索。最终通过 TOR 嵌入完成物理先验的隐式传递。

EIEA 与几何增强主干共享模型底层结构，共同构成 XEmbodied 的完整具身感知能力。

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
```

### 3. EIEA 依赖（仅 EIEA 推理需要）

```bash
pip install peft==0.14.0 transformers==4.49.0
pip install aenum sentencepiece protobuf
```

## 快速开始：路径配置

运行推理脚本前需配置模型路径，仓库默认采用相对路径规划，仅需更新指定占位路径即可。

### 所需模型与权重说明
| 项目 | 默认位置 | 重要说明 |
|------|----------|----------|
| Qwen3-VL-30B-A3B-Instruct | `Qwen/Qwen3-VL-30B-A3B-Instruct` | 基座模型，支持自动下载 |
| VGGT 融合基础文件 | `xembodied/3DA/my_qwen3_vggt_xattnv2/base_ckpt/` | **仅为未训练的权重合并文件，不可直接用于3DA推理** |
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

### 方式一：几何增强主干推理（VGGT 3D 感知）
使用 `tools/infer_debug/infer_demo_show.py`，基于 MS-Swift 调用 XEmbodied 几何增强主干：

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

```

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
