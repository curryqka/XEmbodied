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

> **Authors:**
>
> Kangan Qian<sup>✉,\*</sup>, ChuChu Xie<sup>\*</sup>, Yang Zhong<sup>\*</sup>, Jingrui Pang, Siwen Jiao, Sicong Jiang, Zilin Huang, Yunlong Wang, Kun Jiang<sup>†</sup>, Mengmeng Yang, Hao Ye✉,<sup>†</sup>, Guanghao Zhang, Hangjun Ye, Guang Chen, Long Chen, Diange Yang<sup>†</sup>
>
> **Affiliations:**
>
> Tsinghua University &nbsp; Automotive and Robotics, Xiaomi Corporation &nbsp; National University of Singapore &nbsp; McGill University &nbsp; University of Wisconsin–Madison
>
> ✉ Project Leader &nbsp; \* Equal contribution &nbsp; † Corresponding author

Official implementation of **XEmbodied: A Foundation Model with Enhanced Geometric and Physical Cues for Large-Scale Embodied Environments** ([arXiv 2604.18484](https://arxiv.org/abs/2604.18484)).

We present XEmbodied, a cloud-side foundation model tailored for autonomous driving and robotics VQA within data closed-loop systems. XEmbodied achieves robust large-scale scenario mining and annotation by jointly enhancing domain semantics and 3D understanding while preserving general capabilities. Central to our design is a **3D Adapter (3DA)**, a geometric connector that injects representations from a 3D foundation model into the MLLM via dedicated 3D tokens. Unlike approaches that merely append depth or point clouds as auxiliary inputs, 3DA explicitly aligns 3D structure with language reasoning to endow endogenous 3D competence. We further propose an **Efficient Image-Embodied Adapter (EIEA)**, which distills raw embodied tool outputs into compact token summaries and seamlessly reinserts them into the MLLM context, circumventing the interpretability burden on LLMs.

**Key Contributions:**
- Present XEmbodied, a cloud-side embodied closed-loop VQA generalist fusing intrinsic geometric representations with physical cue interaction.
- Propose 3DA and EIEA-powered implicit physical cue alignment for adaptive geometric priors injection and efficient physical-augmented reasoning.
- Develop a progressive domain curriculum for robust adaptation with reduced forgetting, validated on 18 benchmarks to demonstrate consistent embodied understanding gains.

## News
- 2026/04/27: XEmbodied code officially released
- 2026/04/20: XEmbodied paper available on arXiv

## Results

### Table 1: Spatial & 3D Understanding
The best results among the listed models are **bolded**.

| Model | Ego3DBench ACC | Ego3DBench RMSE | SURDS | VLADBench | STRIDE-QA |
|---|---|---|---|---|---|
| **Proprietary Models** | | | | | |
| GPT-4o [67] | 52.70 | 19.20 | 13.31 | 56.00 | 14.70 |
| Gemini-1.5 [82] | 53.50 | 19.62 | 32.77 | 54.23 | - |
| **Open-Source Models** | | | | | |
| Qwen2.5-VL-7B [5] | 41.10 | 30.36 | 12.61 | 50.75 | 1.41 |
| Qwen2.5-VL-32B [5] | 57.30 | 15.87 | 38.82 | 61.33 | 6.58 |
| Qwen3-VL-A3B-30B [4] | 53.15 | 13.84 | 38.69 | 59.81 | 8.69 |
| Qwen3-VL-32B [4] | **59.93** | 15.70 | 41.53 | 61.64 | 6.00 |
| **Spatial Models** | | | | | |
| UniVG-R1-7B [6] | 46.82 | 53.85 | 1.96 | 46.51 | 4.61 |
| PR1-OCR-2B [101] | 38.88 | 11.60 | 13.81 | 44.65 | 8.18 |
| PR1-Detection-3B [101] | 36.74 | 62.61 | 33.84 | 56.08 | 4.35 |
| PR1-Counting-2B [101] | 40.32 | 51.95 | 13.69 | 48.52 | 1.94 |
| PR1-Grounding-2B [101] | 39.54 | 12.34 | 13.96 | 43.67 | 7.15 |
| **Embodied Models** | | | | | |
| DriveMM [36] | 49.73 | 12.68 | 8.73 | 47.45 | 1.12 |
| Cosmos-R1 [2] | 45.62 | 23.41 | 10.05 | 55.93 | 1.03 |
| Mimo-Embodied [29] | 53.62 | 16.57 | 24.04 | 55.82 | 7.90 |
| **Our Model** | | | | | |
| XEmbodied (Best) | 55.28 | **9.25** | **83.83** | **68.61** | **27.76** |

### Table 2: Semantic & Reasoning
The best results among the listed models are **bolded**.

| Model | DriveBench | DriveLMM-o1 | MapLM-v2 | LingoQA | Omnidrive |
|---|---|---|---|---|---|
| **Proprietary Models** | | | | | |
| GPT-4o [67] | 47.97 | 57.84 | 57.81 | 53.58 | 4.54 |
| Gemini-1.5 [82] | - | - | 58.44 | 61.30 | 2.73 |
| **Open-Source Models** | | | | | |
| Qwen2.5-VL-7B [5] | 43.29 | 37.81 | 55.10 | 53.20 | 2.53 |
| Qwen2.5-VL-32B [5] | 53.06 | 55.04 | 57.47 | 43.70 | 0.00 |
| Qwen3-VL-A3B-30B [4] | 46.22 | 57.17 | 53.63 | 47.00 | 0.31 |
| Qwen3-VL-32B [4] | 51.70 | 55.29 | 46.77 | 54.30 | 0.00 |
| **Spatial Models** | | | | | |
| UniVG-R1-7B [6] | 48.75 | 51.42 | 39.13 | 39.20 | 1.20 |
| PR1-OCR-2B [101] | 50.73 | 47.68 | 44.05 | 48.50 | 4.07 |
| PR1-Detection-3B [101] | 41.35 | 44.81 | 62.47 | 56.08 | 0.10 |
| PR1-Counting-2B [101] | 39.72 | 50.94 | 47.43 | 48.52 | 2.00 |
| PR1-Grounding-2B [101] | 52.03 | 53.83 | 51.00 | 50.30 | 4.85 |
| **Embodied Models** | | | | | |
| DriveMM [36] | 44.50 | 65.91 | 52.18 | 37.70 | 1.10 |
| Cosmos-R1 [2] | 35.80 | 54.66 | 55.33 | 52.70 | 1.23 |
| Mimo-Embodied [29] | 52.95 | 40.31 | 65.23 | **68.60** | 4.90 |
| **Our Model** | | | | | |
| XEmbodied (Best) | **53.18** | **77.01** | **78.55** | 65.70 | **25.43** |

### Table 3: Embodied & Affordance
The best results among the listed models are **bolded**.

| Model | Affordance-2K | Robo-Afford | Cosmos-R1 | Embodied-R1 | RoboRefitBench | VABench-Point | Where2place |
|---|---|---|---|---|---|---|---|
| **Proprietary Models** | | | | | | | |
| GPT-4o [67] | 2.70 | 3.80 | 67.40 | 0.35 | 13.40 | 0.40 | 0.44 |
| Gemini-1.5 [82] | 5.19 | 4.20 | 71.10 | - | 35.50 | 0.80 | 0.89 |
| **Open-Source Models** | | | | | | | |
| Qwen2.5-VL-7B [5] | 8.70 | 3.00 | 72.50 | 0.25 | 76.40 | 0.75 | 0.90 |
| Qwen2.5-VL-32B [5] | 9.10 | 3.25 | 72.50 | 0.00 | 77.60 | 1.45 | 1.25 |
| Qwen3-VL-A3B-30B [4] | 8.85 | 3.65 | 74.50 | 0.05 | **83.95** | 1.10 | 1.51 |
| Qwen3-VL-32B [4] | 6.90 | 4.00 | 75.50 | 0.05 | 83.15 | 1.84 | 1.50 |
| **Embodied Models** | | | | | | | |
| Cosmos-R1 [2] | 6.63 | 1.95 | 72.00 | 0.05 | 34.75 | 0.35 | 0.45 |
| Mimo-Embodied [29] | 8.85 | 3.25 | **81.00** | 0.00 | 78.35 | 1.85 | 1.25 |
| **Our Model** | | | | | | | |
| XEmbodied (Best) | **78.50** | **4.35** | 76.00 | **3.80** | 87.15 | **3.50** | **2.30** |

## TODO

- [x] Release model code
- [x] Release model weights
- [x] Release inference code
- [ ] Release dataset download
- [ ] Release evaluation code
- [ ] Release training code

## Finetuned Models

| Model | Base Model | Weights |
|-------|-----------|---------|
| XEmbodied | Qwen3-VL-30B-A3B-Instruct | [Kaggle](https://www.kaggle.com/models/zhongyangtony/xembodied/) |

## Datasets

XEmbodied is trained on a diverse set of embodied and general VQA datasets:

| Category               | Datasets                                                                 |
|------------------------|--------------------------------------------------------------------------|
| Driving VQA            | LingoQA, SURDS, MapLM, DriveVQA, DriveLMM-o1, Omnidrive, BDD100K         |
| General VQA            | RefCOCO, Flickr, VQAv2, GQA                                              |
| Robot Manipulation     | RoboRefit, Cosmos-R1, RoboVQA                                            |
| 3D / Spatial Reasoning | Visual Trace, VSI-590K, SPAR-7M, OpenSpaces                             |
| Embodied Affordance    | Part Affordance                                                          |

## Project Structure

```
BaseModel-open/
├── xembodied/
│   ├── 3DA/
│   │   └── my_qwen3_vggt_xattnv2/         # Qwen3-VL-MoE-VGGT (3DA geometric connector)
│   │       ├── modeling_qwen3_vl_moe_vggt.py  # Core geometric-enhanced model definition
│   │       ├── configuration_qwen3_vl_moe.py
│   │       ├── modular_qwen3_vl_moe.py
│   │       ├── plugin/
│   │       │   ├── qwen3_vl_moe_vggt_register.py  # MS-Swift custom model registration
│   │       │   └── save_qwen3_vl_moe_vggt.py      # Merge VGGT weights into fused checkpoint
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
    ├── eval_tools/                          # Evaluation aggregation scripts
    ├── vis/                                 # Result visualization
    ├── data_statistics/                     # Data statistics & filtering
    ├── build_model/                         # Model build/save/debug scripts
    └── reward_debug/                        # Reward model debugging
```

## Model Architecture

XEmbodied is a unified foundation model framework for autonomous driving and embodied intelligence, where the geometric enhancement backbone (3DA) and physical cue injection paradigm (EIEA) work in deep synergy. Both are built on Qwen3-VL-30B-A3B-Instruct:

### 1. 3DA: Geometric Spatial Enhancement Backbone

The 3D Adapter (3DA) is a geometric connector that injects representations from a 3D foundation model (VGGT) into the MLLM via dedicated 3D tokens. Unlike approaches that merely append depth or point clouds as auxiliary inputs, 3DA explicitly aligns 3D structure with language reasoning to endow endogenous 3D competence, enabling the VLM to natively understand 3D geometric structure in embodied scenes without relying on external depth models.

Key components:

- **VGGTSpatialEncoder**: Loads the VGGT aggregator to extract multi-layer spatial features from images. Unnecessary heads (camera/point/depth/track) are removed to reduce memory consumption.
- **VGGTEmbeddingMerger**: Aligns and fuses 3D spatial tokens with native 2D visual tokens through bilinear interpolation, token merging, RMSNorm + MLP projection, and Cross-Attention, injecting 3D geometric cues into the vision-language model.

Registered with MS-Swift for one-stop fine-tuning and inference, serving as the foundational carrier of XEmbodied's geometric perception capability.

### 2. EIEA: Efficient Image-Embodied Adapter

While prior work leverages embodied-specific tools (e.g., occupancy grids, 3D boxes, HD map cues) to explicitly guide VQA reasoning, such explicit tool invocation suffers from low efficiency and poor alignment between tool outputs and language reasoning processes. To address these limitations, EIEA distills raw tool outputs into compact token summaries via TOR (Textual Object Rationale) embeddings, which are then seamlessly reinserted into the MLLM context via `masked_scatter`, circumventing the interpretability burden on LLMs.

EIEA shares the model's underlying structure with 3DA, together forming XEmbodied's complete embodied perception capability.

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
pip install transformers==4.49.0
```

### 3. EIEA Dependencies (for EIEA inference)

```bash
pip install peft==0.14.0
pip install aenum sentencepiece protobuf
```

## Quick Start: Path Configuration

Before running any inference script, configure the paths to your local model checkpoints. The repository uses relative paths by default, so you only need to update the specified placeholder paths.

### Required Models & Weights

| Item | Default Location | Notes |
|------|------------------|-------|
| Qwen3-VL-30B-A3B-Instruct | `Qwen/Qwen3-VL-30B-A3B-Instruct` | Base model, supports auto-download |
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

### Method 1: 3DA Geometric Enhancement Inference

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
