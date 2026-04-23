# XEmbodied
Official implementation of XEmbodied: A Foundation Model with Enhanced Geometric and Physical Cues for Large-Scale Embodied Environments.

<div align="center">
<h1><b>XEmbodied</b></h1>
</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2604.18484-B31B1B.svg?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.18484)

</div>

<div align="center">

**[<a href="README_CN.md">中文版</a>]**
**[<a href="https://www.kaggle.com/models/zhongyangtony/xembodied/">Model Weights</a>]**

</div>

Official implementation of **XEmbodied: A Foundation Model with Enhanced Geometric and Physical Cues for Large-Scale Embodied Environments** ([arXiv 2604.18484](https://arxiv.org/abs/2604.18484)).

This repository provides the open-source code for XEmbodied, a unified foundation model framework for autonomous driving and embodied intelligence. It synergistically integrates **VGGT 3D spatial tokens** (geometric perception) and **TOR (Textual Object Rationale) embeddings** (physical prior injection) into the Qwen3-VL-MoE architecture, enhancing spatial cognition, physical interaction reasoning, and scene understanding from both geometric structure and physical constraint dimensions.

## News
- 2026/04/27: XEmbodied code officially released
- 2026/04/20: XEmbodied paper available on arXiv

## Project Structure

```
BaseModel-open/
├── xembodied/
│   ├── 3DA/
│   │   └── my_qwen3_vggt_xattnv2/         # Qwen3-VL-MoE-VGGT (Cross-Attention geometric fusion backbone)
│   │       ├── modeling_qwen3_vl_moe_vggt.py  # Core geometric-enhanced model definition
│   │       ├── configuration_qwen3_vl_moe.py
│   │       ├── modular_qwen3_vl_moe.py
│   │       ├── plugin/
│   │       │   ├── qwen3_vl_moe_vggt_register.py  # MS-Swift custom model registration
│   │       │   └── save_qwen3_vl_moe_vggt.py      # Merge VGGT weights into fused base checkpoint
│   │       └── vggt/                        # VGGT backbone code
│   ├── EIEA/
│   │   ├── infer_with_EIEA_3D.py            # EIEA physical cue injection inference pipeline
│   │   ├── models/
│   │   │   ├── qwen3_vl_vggt_register.py    # Registration with TOR physical embedding auto-injection
│   │   │   ├── qwen3_vl_moe_vggt.py         # Model definition with physical-geometric dual enhancement
│   │   │   └── ...
│   │   ├── tor_embeds_cache_fb.pt           # Pre-cached physical prior TOR embeddings
│   │   └── tor_embeds_cache.pt
│   ├── infer_demo.py                        # Basic VLM inference entry
│   └── infer_demo_show.py                   # Visual inference demo script
└── tools/                                   # Data processing & evaluation toolkit
    ├── convert_*.py                         # Dataset conversion scripts
    ├── infer_debug/
    │   ├── infer_demo_show.py               # Inference with MS-Swift engine
    │   ├── option_parser.py                 # Option parsing utilities
    │   ├── filter_think_format.py           # Think-tag filtering for R1-type models
    │   └── box_match.py                     # Box format regex matching
    ├── agent_tools/                         # Agent tool-call data generation
    ├── eval_tools/                          # Evaluation aggregation scripts
    ├── vis/                                 # Result visualization
    ├── data_statistics/                     # Data statistics & filtering
    ├── build_model/                         # Model build/save/debug scripts
    └── reward_debug/                        # Reward model debugging
```

## Model Architecture

XEmbodied is a **unified foundation model framework for autonomous driving and embodied intelligence**, where the geometric enhancement backbone and physical cue injection paradigm work in deep synergy. Both are built on Qwen3-VL-30B-A3B-Instruct:

### 1. Qwen3-VL-MoE-VGGT (Geometric Spatial Enhancement Backbone)

As the core geometric perception branch of XEmbodied, this model fuses VGGT 3D spatial tokens into the Qwen3-VL architecture via Cross-Attention, enabling native 3D geometric structure understanding for embodied scenes without relying on external depth models.

Key components:

- **VGGTSpatialEncoder**: Loads the VGGT aggregator to extract multi-layer spatial features from images. Unnecessary heads (camera/point/depth/track) are removed to reduce memory consumption.
- **VGGTEmbeddingMerger**: Aligns and fuses 3D spatial tokens with native 2D visual tokens through bilinear interpolation, token merging, projection, and cross-attention, injecting 3D geometric cues into the vision-language model.

Registered with MS-Swift for one-stop fine-tuning and inference, serving as the foundational carrier of XEmbodied's geometric perception capability.

### 2. EIEA Physical Cue Injection Paradigm

EIEA (External Implicit Embodied Agent) is XEmbodied's **physical prior injection mechanism** designed for embodied scenarios. Its core is injecting scene physical constraints and object interaction rationality into the model, ultimately completing the implicit transfer of physical priors through TOR embeddings.

EIEA shares the model's underlying structure with the geometric enhancement backbone, together forming XEmbodied's complete embodied perception capability.

## Environment Setup

### 1. Base Environment

```bash
conda create -n xembodied python=3.10
conda activate xembodied

# Install PyTorch
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
```

### 2. MS-Swift (Training & Inference Framework)

```bash
pip install ms-swift
pip install qwen_vl_utils==0.0.14
pip install trl -U
pip install vllm -U
```

### 3. EIEA Dependencies (for EIEA inference only)

```bash
pip install peft==0.14.0 transformers==4.49.0
pip install aenum sentencepiece protobuf
```

## Quick Start: Path Configuration

Before running any inference script, configure the paths to your local model checkpoints. The repository uses relative paths by default, so you only need to update the specified placeholder paths.

### Required Models & Weights

| Item | Default Location | Notes |
|------|------------------|-------|
| Qwen3-VL-30B-A3B-Instruct | `Qwen/Qwen3-VL-30B-A3B-Instruct` | Base model, supports auto-download |
| VGGT fused base files | `xembodied/3DA/my_qwen3_vggt_xattnv2/base_ckpt/` | **Weight merge only, not directly usable for 3DA inference** |
| Trained model weights | Kaggle Model Scope | Full trainable weights: https://www.kaggle.com/models/zhongyangtony/xembodied/ |
| VGGT register file | `xembodied/3DA/.../qwen3_vl_moe_vggt_register.py` | MS-Swift model registration file |
| EIEA register file | `xembodied/EIEA/models/qwen3_vl_vggt_register.py` | Registration with physical embedding injection |
| TOR embedding cache | `xembodied/EIEA/tor_embeds_cache_fb.pt` | Built-in pre-cached file, no extra model download needed |

### Setup Steps

1. **Download the base VLM**: Download [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) or let MS-Swift auto-download it.
2. **Get trainable model weights**: Download the trained XEmbodied weights from [Kaggle Model Scope](https://www.kaggle.com/models/zhongyangtony/xembodied/).
3. **EIEA physical embedding**: The repository includes pre-cached TOR embedding files — no extra pretrained model downloads required.
4. **Update script paths**: Modify the model path in inference scripts to your local weight path. Keeping the default directory structure allows automatic relative path adaptation.

## Inference

### Method 1: Geometric Enhancement Backbone Inference (VGGT 3D Perception)

Use `xembodied/infer_demo.py` to run XEmbodied's geometric enhancement backbone via MS-Swift:

```bash
cd XEmbodied
python ./xembodied/infer_demo.py
```

Script configuration:

```python
# Enable XEmbodied geometric enhancement model
USE_CUSTOM_MODEL = True

# Custom model config (use Kaggle-downloaded trained weights)
CUSTOM_MODEL_CFG = {
    "model_path": "Path to Kaggle-downloaded model weights",
    "model_type": "qwen3_vl_moe_vggt",
    "register_file": "<path-to-register-file>",
}

# Native model config
NATIVE_MODEL_CFG = {
    "model_path": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "model_type": "qwen3_vl_moe",
}

# Inference parameters
INFER_CFG = {
    "max_tokens": 512,
    "temperature": 0,
    "user_prompt": "Your prompt here",
    "image_paths": ["<image_path>"]
}
```

### Method 2: EIEA Physical-Geometric Dual Enhancement Inference

Use `xembodied/EIEA/infer_with_EIEA_3D.py` for the full physical cue injection inference pipeline:

```bash
cd xembodied/EIEA
python infer_with_EIEA_3D.py
```

This script implements physical-geometric dual-cue synergistic inference. Built-in pre-cached TOR embeddings eliminate the need for downloading extra pretrained models, automatically completing scene physical prior injection and 3D geometric perception inference.

## Tools

| Directory/Script | Description |
|-----------------|-------------|
| `convert_*.py` | Convert embodied datasets (NuScenes, OmniDrive, LingoQA, etc.) to training JSONL format |
| `parse_*.py` | Parse raw evaluation outputs to standardized format |
| `infer_debug/` | Inference debug and format filtering tools |
| `eval_tools/` | Embodied scene evaluation result aggregation |
| `vis/` | Interaction and perception result visualization |
| `data_statistics/` | Data statistics, filtering, and sampling |
| `build_model/` | Model loading, saving, and debugging |
| `reward_debug/` | GRPO reward model debugging |

## Citation

If this project is useful in your work, we'd appreciate a citation:

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
