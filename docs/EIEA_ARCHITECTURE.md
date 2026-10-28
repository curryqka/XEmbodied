# EIEA Architecture: Forward Pass and Design Clarifications

This document clarifies the EIEA (Efficient Image-Embodied Adapter) forward pass, addressing reviewer questions about Mamba input ordering, token count, and dimensional consistency.

## Mamba-2 Input Sequence

The Mamba-2 interpreter processes the following sequence in a single causal scan:

```
[image_proj(IE); depth_proj(DE); box_tokens(BX); bev_tokens(BV); TOR_queries(Q); text_ids(T)]
```

| Segment | Source | Shape | Description |
|---------|--------|-------|-------------|
| `IE` | CLIP ViT-L last hidden states | `[1, N_img, 1024]` | Front-view image features |
| `DE` | SigLIP-SO400M last hidden states | `[1, N_dep, 1152]` | Depth map features (DepthPro) |
| `BX` | Box encoder output | `[1, N_box, 768]` | 3D box tokens (class emb + geometric params → MLP) |
| `BV` | BEV encoder output | `[1, 64, 768]` | BEV grid tokens (CNN → 8×8 spatial pool → Linear) |
| `Q` | Learnable TOR queries | `[1, 10, 768]` | 10 latent reasoning tokens |
| `T` | Token embeddings | `[1, L_text, 768]` | Question token IDs via Mamba embedding table |

All segments are projected to `d_mamba = 768` before concatenation. The full sequence length is `N_img + N_dep + N_box + 64 + L_text`.

## Retained Positions

Only the hidden states at the 10 TOR query positions are extracted:

```
tor_start = N_img + N_dep + N_box
tor_end   = tor_start + 10
tor_hidden = hidden_states[:, tor_start:tor_end, :]
```

These 10 positions are placed **after** all image/depth/box tokens and **before** the question text, so their causal states have already aggregated all physical-cue evidence. No bidirectional scan is needed.

## Projection to VLM Space

The TOR hidden states (d_mamba = 768) are projected to d_VLM via:

```
Z_phy = Linear(GELU(Linear(RMSNorm(tor_hidden))))   # [10, d_VLM]
```

For the 30B model, `d_VLM = 2048` (matching Qwen3-VL-30B-A3B's hidden size). The `10 × 3584` cache in the released code is from an earlier Qwen2.5-VL-7B prototype (7B hidden size = 3584).

## TOR Token Injection

The 10 projected cue tokens are injected into the frozen VLM's input embeddings:

```python
mask = input_ids == tor_token_id
inputs_embeds = inputs_embeds.masked_scatter(
    mask.unsqueeze(-1).expand_as(inputs_embeds), tor_embeds)
```

This is equivalent to replacing the embedding of 10 designated `<TOR>` tokens with the cue representations. The VLM backbone is frozen; only the cue encoder is trained.

## Token Count Clarification

The paper and code use **10 latent reasoning tokens**, not 64. The "64" in the supplementary material is a typographical error (the supplementary text also states "insert 10 latent reasoning tokens").

## Dimensional Clarification

The `(1, 10, 3584)` demo cache in the released code is from the early Qwen2.5-VL-7B prototype. The 7B hidden size is 3584, hence the cache dimension. The released 30B model uses d_VLM = 2048, and the cue encoder output is projected accordingly. The `(1, 10, 3584)` cache is **not** the final EIEA output format.
