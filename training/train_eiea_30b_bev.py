#!/usr/bin/env python3
"""Train EIEA cue encoder (MIDI + box + BEV) with frozen 30B, answer CE.

Architecture:
  image → CLIP → image_proj → ┐
  depth → SigLIP → depth_proj → ├→ Mamba-2 → TOR hidden → φ_R → 10×2048
  boxes → BoxEncoder → ────────┤                                ↓
  BEV → BEVEncoder → ─────────┘                          masked_scatter into 30B
  question → Mamba embeds → ─┘                                ↓
                                                         answer CE loss
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ssr_eiea_common import TOR, load_midi, load_visual_encoders
from ssr.utils.prompt import insert_tor
from train_e2e import E2EEncoder

MODEL_DIR = os.environ.get("XEMB_MODEL_DIR", "/dev/shm/models/xembodied-v0")
EIEA_DIR = "/home/kaneqian/XEmbodied/code/xembodied-code - AAAI/xembodied/EIEA/models"
PROJ = ("/home/kaneqian/XEmbodied/code/xembodied-code - AAAI/xembodied/3DA/"
        "my_qwen3_vggt_xattnv2")
OUT = "/home/kaneqian/XEmbodied/results/eiea_box"
TOR_ID = 151643
TOR_STR = "<|endoftext|>"
NUM_BOX_FEATS = 10


class EIEAEncoder(nn.Module):
    """EIEA cue encoder: MIDI + box + BEV encoders → 10 × d_VLM tokens."""

    def __init__(self, midi, hidden_size=768, d_vlm=2048,
                 use_boxes=True, use_bev=True, bev_ch=5, num_tor=10):
        super().__init__()
        self.midi = midi
        self.hidden_size = hidden_size
        self.d_vlm = d_vlm
        self.num_tor = num_tor
        self.use_boxes = use_boxes
        self.use_bev = use_bev

        self.box_enc = nn.Sequential(
            nn.Linear(11, 512), nn.GELU(), nn.Linear(512, hidden_size))
        self.bev_enc = nn.Sequential(
            nn.Conv2d(bev_ch, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(), nn.Linear(128 * 64, hidden_size))
        # Mamba output (hidden_size) → VLM space (d_vlm)
        self.proj = nn.Sequential(
            nn.RMSNorm(hidden_size, eps=1e-6),
            nn.Linear(hidden_size, d_vlm), nn.GELU(),
            nn.Linear(d_vlm, d_vlm))
        # Cross-attention: 10 learnable queries attend to all cue tokens
        self.queries = nn.Parameter(torch.randn(num_tor, hidden_size) * 0.02)
        self.cross_attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.out_norm = nn.RMSNorm(hidden_size, eps=1e-6)

    def forward(self, image_embeds, depth_embeds, boxes, bev, text_ids, tor_token_id):
        """All inputs on same device.
        Returns: [1, num_tor, d_vlm] cue tokens.
        """
        m = self.midi
        ie = m.image_proj(image_embeds)
        de = m.depth_proj(depth_embeds)
        te = m.mamba.get_input_embeddings()(text_ids)

        parts = [ie, de]
        if self.use_boxes and boxes is not None and boxes.shape[0] > 0:
            cls = self.box_enc[0](boxes[:, 0].long()) if False else None
            # boxes: [N, 11] → class_emb + 10 numeric
            parts.append(self.box_enc(boxes[:, :11]).unsqueeze(0))
        if self.use_bev and bev is not None:
            bv = self.bev_enc(bev.unsqueeze(0))  # [1, hidden]
            parts.append(bv.unsqueeze(1))  # [1, 1, hidden]

        parts.append(te)
        seq = torch.cat(parts, dim=1)
        mask = torch.ones(1, seq.shape[1], dtype=torch.long, device=seq.device)
        out = m.mamba(inputs_embeds=seq, attention_mask=mask, output_hidden_states=True)
        hidden = out.hidden_states[-1]

        # Extract hidden states at TOR positions (last 10 before text)
        # TOR queries were appended before text: [..., tor_queries(10), text(L)]
        tor_offset = seq.shape[1] - te.shape[1]  # text comes after TOR queries in our setup
        # Actually: we placed TOR queries between structured cues and text.
        # Let's use the positions where text_ids == tor_token_id.
        tor_mask = (text_ids == tor_token_id)
        tor_hidden = hidden[:, tor_mask[0], :]  # [10, hidden]

        cue = self.proj(tor_hidden.squeeze(0) if tor_hidden.dim() == 3 else tor_hidden)
        # Cross-attention: TOR queries attend to all cue tokens for refinement
        q = self.queries.unsqueeze(0).to(cue.dtype)
        kv = cue.unsqueeze(0)
        refined, _ = self.cross_attn(self.out_norm(q), self.out_norm(kv), self.out_norm(kv))
        return (cue + refined).unsqueeze(0)  # [1, num_tor, d_vlm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr_midi", type=float, default=2e-5)
    ap.add_argument("--lr_new", type=float, default=2e-4)
    ap.add_argument("--data", default=os.path.join(OUT, "train_cues.json"))
    ap.add_argument("--bev_dir", default=os.path.join(OUT, "bev"))
    ap.add_argument("--save", default=os.path.join(OUT, "eiea_30b_bev.pt"))
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, PROJ := "/home/kaneqian/XEmbodied/code/xembodied-code - AAAI/xembodied/3DA/my_qwen3_vggt_xattnv2")
    sys.path.insert(0, EIEA := "/home/kaneqian/XEmbodied/code/xembodied-code - AAAI/xembodied/EIEA/models")
    from transformers import AutoConfig, AutoProcessor
    from qwen_vl_utils import process_vision_info
    spec = importlib.util.spec_from_file_location(
        "eiea_modeling", os.path.join(EIEA, "qwen3_vl_moe_vggt.py"))
    eiea_mod = importlib.util.module_from_spec(spec)
    sys.modules["eiea_modeling"] = eiea_mod
    spec.loader.exec_module(eiea_mod)
    C = eiea_mod.Qwen3VLMoeVGGTForConditionalGeneration

    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    n_layers = cfg.text_config.num_hidden_layers
    dm = {f"model.language_model.layers.{i}": (0 if i < 22 else 1) for i in range(n_layers)}
    dm.update({"model.language_model.embed_tokens": 0, "model.language_model.norm": 1,
               "model.language_model.rotary_emb": 0, "model.visual": 0,
               "spatial_encoder": 0, "connector": 0, "lm_head": 1})
    cfg.vggt_device = "cuda:0"; cfg.vggt_dtype = "bfloat16"
    model = C.from_pretrained(MODEL_DIR, config=cfg, dtype=torch.bfloat16,
                              device_map=dm, trust_remote_code=True)
    model.train(); model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in model.parameters():
        p.requires_grad_(False)
    proc = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    tps = int(cfg.vision_config.temporal_patch_size)

    midi, mtok = load_midi(str(dev)); midi.train()
    for p in midi.parameters():
        p.requires_grad_(True)
    try:
        midi.mamba.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        pass
    cp, clip, sp, sig = load_visual_encoders(str(dev))
    for m in (clip, sig):
        for p in m.parameters():
            p.requires_grad_(False)

    # Sub-encoders (box + BEV)
    box_enc = nn.Sequential(nn.Linear(11, 512), nn.GELU(), nn.Linear(512, 768)).to(dev)
    bev_enc = nn.Sequential(
        nn.Conv2d(5, 32, 3, 2, 1), nn.GELU(),
        nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
        nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(),
        nn.AdaptiveAvgPool2d((8, 8)),
        nn.Flatten(), nn.Linear(128 * 64, 768)).to(dev)
    proj = nn.Sequential(
        nn.RMSNorm(768, eps=1e-6), nn.Linear(768, 2048),
        nn.GELU(), nn.Linear(2048, 2048)).to(dev)
    queries = nn.Parameter(torch.randn(10, 768, device=dev) * 0.02)
    cross_attn = nn.MultiheadAttention(768, 8, batch_first=True).to(dev)
    out_norm = nn.RMSNorm(768, eps=1e-6).to(dev)

    trainable = (list(midi.parameters()) + list(box_enc.parameters()) +
                 list(bev_enc.parameters()) + list(proj.parameters()) +
                 list(cross_attn.parameters()) + list(out_norm.parameters()) +
                 [queries])
    opt = torch.optim.AdamW(trainable, lr=args.lr_new)
    # MIDI uses lower lr
    for g in opt.param_groups:
        pass  # single group for simplicity; can split if needed

    rows = json.load(open(args.data))
    bevs = {i: torch.from_numpy(np.load(os.path.join(args.bev_dir, f"train_{i}.npy"))).float().to(dev)
            for i in range(min(args.steps, len(rows))) if os.path.exists(os.path.join(args.bev_dir, f"train_{i}.npy"))}

    t0 = time.time()
    for i in range(args.steps):
        r = rows[i % len(rows)]
        img = Image.open(r["images"][1]).convert("RGB")
        dep = np.load(r["depth"]) if "depth" in r and os.path.exists(r["depth"]) else None

        with torch.no_grad():
            ie = clip(**cp(images=img, return_tensors="pt").to(dev)).last_hidden_state.float()
            if dep is not None:
                de = sig(**sp(images=dep, return_tensors="pt").to(dev)).last_hidden_state.float()
            else:
                de = torch.zeros(1, 729, 1152, device=dev)

        ids = mtok(r["question"] + insert_tor("", 10), add_special_tokens=False, return_tensors="pt")
        boxes = torch.tensor(r.get("boxes", []), dtype=torch.float32, device=dev) if r.get("boxes") else torch.zeros(0, 11, device=dev)
        bev = bevs.get(i)

        # Build Mamba sequence
        ie_p = midi.image_proj(ie)
        de_p = midi.depth_proj(de)
        te = midi.mamba.get_input_embeddings()(ids.to(dev))

        parts = [ie_p, de_p]
        if boxes.shape[0] > 0:
            parts.append(box_enc(boxes).unsqueeze(0))
        if bev is not None:
            bv = bev_enc(bev.unsqueeze(0))  # [1, 768]
            parts.append(bv.unsqueeze(1))
        parts.append(te)
        seq = torch.cat(parts, 1)
        mask = torch.ones(1, seq.shape[1], dtype=torch.long, device=dev)
        mamba_out = midi.mamba(inputs_embeds=seq, attention_mask=mask, output_hidden_states=True)
        hidden = mamba_out.hidden_states[-1]

        # Extract TOR positions (after structured cues, in text section)
        tor_hidden = hidden[:, -10:, :]  # last 10 tokens are TOR queries
        cue = proj(tor_hidden.squeeze(0))  # [10, 2048]

        # Cross-attention refinement
        q = queries.unsqueeze(0)
        kv = cue.unsqueeze(0)
        refined, _ = cross_attn(out_norm(q), out_norm(kv), out_norm(kv))
        cue_final = (cue + refined).unsqueeze(0).to(torch.bfloat16)  # [1, 10, 2048]

        # VLM forward with cue injection
        user_text = r["question"]
        if r.get("options"):
            user_text += "\nOptions:\n" + "\n".join(r["options"])
        text_with_tor = TOR_STR * 10 + "\n" + user_text + "\nAnswer:"
        user = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": text_with_tor}]}]
        full = proc.apply_chat_template(user + [{"role": "assistant", "content": r["answer"]}],
                                        tokenize=False, add_generation_prompt=False)
        prefix = proc.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
        enc = proc(text=[full], images=[img], return_tensors="pt")
        n_prefix = proc(text=[prefix], images=[img], return_tensors="pt")["input_ids"].shape[1]
        labels = torch.full_like(enc["input_ids"], -100)
        labels[:, n_prefix:] = enc["input_ids"][:, n_prefix:]
        kw = {k: v.to(dev) for k, v in enc.items()
              if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
        tchw = torch.stack([prep(img)] * (int(enc["image_grid_thw"][0, 0]) * tps), 0).to(dev)
        out = model(**kw, image_tchw=[tchw], labels=labels.to(dev),
                    tor_embeds=cue_final, tor_token_id=151643)
        out.loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if i % 10 == 0:
            print(f"{i}/{args.steps} loss={out.loss.item():.4f} grad={float(gn):.3f} "
                  f"{time.time() - t0:.0f}s", flush=True)
        if args.smoke and i + 1 >= args.smoke:
            break

    torch.save({"midi": midi.state_dict(),
                "box_enc": box_enc.state_dict(),
                "bev_enc": bev_enc.state_dict(),
                "proj": proj.state_dict(),
                "cross_attn": cross_attn.state_dict(),
                "out_norm": out_norm.state_dict(),
                "queries": queries}, args.save)
    print("saved", args.save)


if __name__ == "__main__":
    main()
