<div align="center">

# XEmbodied

**A Foundation Model with Enhanced Geometric and Physical Cues for Large-Scale Embodied Environments**

[![arXiv](https://img.shields.io/badge/arXiv-2604.18484-B31B1B.svg)](https://arxiv.org/abs/2604.18484)
[![Official Code](https://img.shields.io/badge/Official%20Code-xiaomi--research-blue)](https://github.com/xiaomi-research/xiaomi_spatialembodied)
[![Model Weights](https://img.shields.io/badge/Model-ModelScope-624aff)](https://www.modelscope.cn/models/xzhong/xembodied-v0)

**[中文版](README_CN.md)**

</div>

## Overview

XEmbodied (released as **Xiaomi-SpatialEmbodied**) is a foundation model for cloud-side embodied VQA in autonomous-driving and robotics data closed loops. It builds on **Qwen3-VL-30B-A3B-Instruct** and adds two complementary interfaces:

- **3DA (3D Adapter)** injects geometric features from a VGGT 3D foundation model into the Qwen visual tokens through a cross-attention residual.
- **EIEA (Efficient Image-Embodied Adapter)** distills occupancy, 3D-box and map cues into compact TOR embeddings, providing physical priors without exposing raw tool traces.
- A **progressive domain curriculum** adapts the model from general multimodal data to spatial and embodied data with reduced forgetting.

The paper reports consistent gains on 18 spatial, semantic, embodied and affordance benchmarks.

## Official resources

The **official repository is the authoritative source** for architecture, installation, model weights, inference and complete usage instructions:

| Resource | Link |
|---|---|
| **Official repository / README** | **https://github.com/xiaomi-research/xiaomi_spatialembodied** |
| Official README (English) | https://github.com/xiaomi-research/xiaomi_spatialembodied/blob/main/README.md |
| Official README (中文) | https://github.com/xiaomi-research/xiaomi_spatialembodied/blob/main/README_CN.md |
| Paper | https://arxiv.org/abs/2604.18484 |
| Model weights | https://www.modelscope.cn/models/xzhong/xembodied-v0 |

> Please follow the official README for environment setup, checkpoints, inference commands, benchmark tables and the latest updates. This repository intentionally keeps only a short overview.

## What this repository contains

This is a companion working codebase used for experiments around the paper:

| Directory | Contents |
|---|---|
| `data/` | Converters and statistics for DriveBench, DriveLMM-o1, Ego3D-Bench, OmniDrive, MapLM-v2, VLAD-Bench, RoboAfford, PartAfford, RoboRefit and others |
| `evaluation/` | Per-benchmark evaluation scripts and a unified benchmark system prompt |
| `training/` | EIEA data pipeline and training entry points (including 3DA/box/BEV variants) |
| `model/` | Model-side utilities used by the training/evaluation scripts |
| `tools/` | Cost benchmarks, paired bootstrap/McNemar statistics and other experiment utilities |
| `docs/` | Architecture, training, normalization, data provenance and benchmark coverage notes |
| `scripts/infer_demo.py` | Small demo entry point |

For the maintained model implementation and canonical instructions, always use the official repository above.

## Citation

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

## Authors

Kangan Qian\*, ChuChu Xie\*, Yang Zhong<sup>✉,\*</sup>, Jingrui Pang, Siwen Jiao, Sicong Jiang, Zilin Huang, Yunlong Wang, Kun Jiang†, Mengmeng Yang, Hao Ye<sup>✉,†</sup>, Guanghao Zhang, Hangjun Ye, Guang Chen, Long Chen, Diange Yang†

Automotive and Robotics, Xiaomi Corporation · Tsinghua University · National University of Singapore · McGill University · University of Wisconsin–Madison

✉ Project Leader · \* Equal contribution · † Corresponding author
