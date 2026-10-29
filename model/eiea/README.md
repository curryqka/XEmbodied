# EIEA: Efficient Image-Embodied Adapter

Reference implementation of the physical-cue branch of XEmbodied. It encodes
3D boxes, a BEV map and a depth map (optionally BEV flow) alongside the
front-view image and question, runs a frozen Mamba-2 interpreter, and injects
`K = 10` compact cue tokens into the frozen Qwen3-VL-30B-A3B context at
designated `<TOR>` positions.

- Architecture / sequence order: [`docs/EIEA_ARCHITECTURE.md`](../../docs/EIEA_ARCHITECTURE.md)
- Training recipe (box + BEV + depth, random masking, noise): [`docs/EIEA_TRAINING.md`](../../docs/EIEA_TRAINING.md)
- Encoder: [`cue_encoder.py`](cue_encoder.py)
- Corruption: [`corruption.py`](corruption.py)
- loaders for the 7B SSR proxy: [`ssr_eiea_common.py`](ssr_eiea_common.py)

## Quick start

```python
import torch
from model.eiea.cue_encoder import EIEAConfig, EIEACueEncoder

encoder = EIEACueEncoder(mamba_model, EIEAConfig(d_vlm=2048, num_tor=10))
cue = encoder(image_embeds, depth_embeds, boxes, bev, flow=None, text_ids)
# cue: [1, 10, 2048] -> inject at the 10 <TOR> positions
```

`mamba_model` is an SSR-MIDI `MambaForCausalLM` (mamba-130m). Use
`EIEACueEncoder.from_midi(midi, cfg)` to copy the MIDI image/depth projections
and Mamba weights.

## Scope of this release

- The **30B EIEA training/eval code** in this repository is a reference
  implementation. It needs the base XEmbodied checkpoint plus the SSR-MIDI,
  CLIP ViT-L and SigLIP checkpoints; see `docs/EIEA_TRAINING.md`.
- The external demo caches (`tor_embeds_cache*.pt`, 10×3584) are **not
  bundled** here and are **not** compatible with the 30B model's 2048-d word
  embeddings; the register now warns and skips auto-injection instead of
  failing with a shape error. Set `XEMB_TOR_CACHE` to a matching cache when
  one is available.
- EIEA checkpoint weights produced by the paper's Stage 4 are not part of the
  public release; this repository ships the architecture, the trainer and the
  evaluation harness.

## Environment variables

| Variable | Meaning |
|----------|---------|
| `SSR_ROOT` | path to the SSR source tree (for `ssr.models.midi` / `ssr.models.vlm`) |
| `SSR_MODEL_ROOT` | root that contains the SSR checkpoints |
| `SSR_MIDI_DIR` | SSR-MIDI-7B checkpoint |
| `SSR_MAMBA_DIR` | mamba-130m-hf checkpoint |
| `SSR_CLIP_DIR` / `SSR_SIGLIP_DIR` | CLIP ViT-L / SigLIP SO400M |
| `SSR_VLM_BASE` / `SSR_VLM_ADAPTER` | Qwen2.5-VL-7B base + SSR LoRA adapter |
| `XEMB_MODEL_DIR` | base XEmbodied 30B checkpoint |
| `XEMB_TOR_CACHE` / `XEMB_TOR_TOKEN_ID` | TOR cache and placeholder token id |
| `VGGT_REPO_DIR` | directory containing the `vggt/` package (defaults to `model/3da`) |
