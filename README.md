# XEmbodied

Code for **XEmbodied: A Foundation Model for Embodied Scene Understanding**.

## Overview

XEmbodied augments a vision-language backbone (Qwen3-VL-30B-A3B) with two complementary adapters:

- **3DA (3D Adapter)**: injects VGGT-extracted multi-view geometric features into the 2D semantic stream via cross-attention.
- **EIEA (Efficient Image-Embodied Adapter)**: distills structured physical cues (BEV maps, motion, 3D boxes) into compact latent tokens via a Mamba-2 interpreter.

## Repository Structure

```
XEmbodied/
├── model/
│   ├── 3da/                    # 3D Adapter (VGGT + cross-attention fusion)
│   │   ├── modeling_qwen3_vl_moe_vggt.py
│   │   ├── configuration_qwen3_vl_moe.py
│   │   ├── qwen3_vl_moe_vggt_register.py
│   │   └── vggt/               # VGGT backbone (Meta, Apache-2.0)
│   └── eiea/                   # EIEA (physical cue token injection)
│       ├── qwen3_vl_moe_vggt.py
│       └── qwen3_vl_vggt_register.py
├── training/                   # 4-stage progressive training
│   ├── sft_ddp_s1.sh           # Stage 1: domain semantic alignment
│   ├── sft_ddp_s2.sh           # Stage 2: 3D geometry alignment
│   ├── sft_ddp_s3.sh           # Stage 3: end-to-end geometric cognition
│   ├── sft_ddp_s4.sh           # Stage 4: EIEA alignment + GRPO
│   └── dataset.py              # dataset registration
├── evaluation/                 # benchmark evaluation scripts
│   ├── eval_drivelmm.py        # DriveLMM-o1
│   ├── eval_drivebench.py      # DriveBench
│   ├── eval_ego3d.py           # Ego3D-Bench
│   ├── ...                     # 18 benchmarks total
│   └── unified_benchmark_system_prompt.py
├── data/                       # data conversion & curation
│   ├── convert_*.py            # per-benchmark data converters
│   ├── data_statistics_val.py  # data quality statistics
│   └── filter_*.py             # filtering utilities
├── tools/
│   └── paired_stats.py         # paired bootstrap / McNemar
└── scripts/
    ├── infer_demo.py           # inference demo
    └── infer.sh                # inference launcher
```

## Requirements

```
transformers>=4.57.1
torch>=2.5.0
qwen-vl-utils>=0.0.14
decord
ms-swift
flash-attn
```

## Model Weights

Download from [ModelScope](https://www.modelscope.cn/models/xzhong/xembodied-v0):

```bash
modelscope download --model xzhong/xembodied-v0 --local_dir ./checkpoints/xembodied-v0
```

## Inference

```bash
bash scripts/infer.sh
```

## Training

Training follows a 4-stage progressive curriculum (details in paper Appendix D):

| Stage | Objective | Trainable | Data |
|-------|-----------|-----------|------|
| 1 | Domain semantic alignment | LLM (ViT + 3DA frozen) | T1 |
| 2 | 3D geometry alignment | 3DA only (backbone frozen) | T2 |
| 3 | End-to-end geometric cognition | 3DA + LoRA (r=64) | T2→T3→T4 |
| 4 | EIEA alignment + GRPO | EIEA + LoRA | T1–T4 (reuse) |

```bash
bash training/sft_ddp_s1.sh  # Stage 1
bash training/sft_ddp_s2.sh  # Stage 2
# ... etc
```

## Evaluation

Each benchmark has a dedicated evaluation script in `evaluation/`. All scripts accept a JSONL inference file and a dataset metadata file.

```bash
python evaluation/eval_ego3dbenchr1.py \
    --infer_file_path results.jsonl \
    --dataset_path ego3d_meta.jsonl \
    --model_name xembodied \
    --output_dir eval_output/
```

## License

Apache License 2.0
