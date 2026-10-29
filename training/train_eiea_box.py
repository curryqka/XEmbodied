#!/usr/bin/env python3
"""Train an EIEA-lite on top of SSR MIDI with box input.

Frozen: SSR-VLM (Qwen2.5-VL-7B + SSR LoRA), CLIP/SigLIP encoders.
Trainable: MIDI (Mamba + projections) + a box encoder.
Loss: answer cross-entropy through the frozen VLM with 10 cue tokens.
"""
import argparse
import json
import os
import re

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ssr_eiea_common import (
    DEPTH_CKPT, TOR, load_midi, load_visual_encoders, load_vlm,
)
from ssr.utils.prompt import insert_tor

OUT = "/home/kaneqian/XEmbodied/results/eiea_box"
NUM_BOX_FEATS = 10  # x,y,z,l,w,h,yaw,vx,vy,dist


class BoxMIDI(nn.Module):
    def __init__(self, midi, hidden=768):
        super().__init__()
        self.midi = midi
        self.class_emb = nn.Embedding(11, 16)
        self.box_mlp = nn.Sequential(
            nn.Linear(16 + NUM_BOX_FEATS, hidden), nn.GELU(),
            nn.Linear(hidden, hidden))

    def forward(self, image_embeds, depth_embeds, boxes, ids, tor_id):
        m = self.midi
        ie = m.image_proj(image_embeds)
        de = m.depth_proj(depth_embeds)
        if boxes.shape[0] > 0:
            be = self.box_mlp(torch.cat([self.class_emb(boxes[:, 0].long()),
                                         boxes[:, 1:1 + NUM_BOX_FEATS]], -1))
        else:
            be = ie.new_zeros((0, ie.shape[-1]))
        te = m.mamba.get_input_embeddings()(ids)
        seq = torch.cat([ie, de, be.unsqueeze(0), te], 1)
        mask = torch.ones(1, seq.shape[1], dtype=torch.long, device=seq.device)
        out = m.mamba(inputs_embeds=seq, attention_mask=mask, output_hidden_states=True)
        hidden = out.hidden_states[-1][:, ie.size(1) + de.size(1) + be.size(0):, :]
        tor = m.tor_proj(hidden[(ids == tor_id)])
        return tor


def _messages(r, answer=None):
    # front view only: boxes are in the ego frame, so the cue encoder carries
    # the spatial evidence; this keeps training affordable on 2x4090.
    content = [{"type": "image", "image": Image.open(r["images"][1]).convert("RGB").resize((256, 256))}]
    content.append({"type": "text", "text": insert_tor("", 10) + "\n" + r["question"]})
    msgs = [{"role": "user", "content": content}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


def prepare_vlm_batch(proc, r, device, add_answer=None):
    from qwen_vl_utils import process_vision_info
    msgs = _messages(r, add_answer)
    prompt = proc.apply_chat_template(msgs, tokenize=False,
                                      add_generation_prompt=(add_answer is None))
    imgs, _ = process_vision_info(msgs)
    enc = proc(text=[prompt], images=imgs, videos=None, padding=True, return_tensors="pt")
    kw = {k: v.to(device) for k, v in enc.items()
          if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
    return kw


def answer_labels(proc, r, input_ids, device):
    """Label the assistant answer: prefix = user turn + assistant header."""
    from qwen_vl_utils import process_vision_info
    msgs = _messages(r, None)
    p = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(msgs)
    enc = proc(text=[p], images=imgs, videos=None, padding=True, return_tensors="pt")
    n_prefix = enc["input_ids"].shape[1]
    labels = torch.full_like(input_ids, -100)
    labels[:, n_prefix:] = input_ids[:, n_prefix:]
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--save", default=os.path.join(OUT, "encoder.pt"))
    ap.add_argument("--data", default=os.path.join(OUT, "train.json"))
    args = ap.parse_args()

    from transformers import AutoProcessor
    rows = json.load(open(args.data))
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)

    vlm = load_vlm(str(dev))
    vlm.train()
    for p in vlm.parameters():
        p.requires_grad_(False)
    vlm.gradient_checkpointing_enable()
    proc = AutoProcessor.from_pretrained(
        os.environ.get("SSR_VLM_BASE", "/dev/shm/models/Qwen2.5-VL-7B-Instruct"))
    proc.tokenizer.add_tokens(TOR, special_tokens=True)
    tid = proc.tokenizer.convert_tokens_to_ids(TOR)

    midi, mtok = load_midi(str(dev))
    for p in midi.parameters():
        p.requires_grad_(True)
    enc = load_visual_encoders(str(dev))
    for m in enc[1::1]:
        if isinstance(m, torch.nn.Module):
            for p in m.parameters():
                p.requires_grad_(False)
    cp, clip, sp, sig = enc
    model = BoxMIDI(midi).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def step(r):
        im = Image.open(r["images"][1]).convert("RGB")
        depth = np.load(r["depth"])
        with torch.no_grad():
            mdtype = next(model.parameters()).dtype
            ie = clip(**cp(images=im, return_tensors="pt").to(dev)).last_hidden_state.float().to(mdtype)
            de = sig(**sp(images=depth, return_tensors="pt").to(dev)).last_hidden_state.float().to(mdtype)
        qids = mtok(r["question"] + insert_tor("", 10), add_special_tokens=False, return_tensors="pt")
        boxes = torch.tensor(r["boxes"], dtype=torch.float32, device=dev) if r["boxes"] else torch.zeros(0, 13, device=dev)
        cue = model(ie, de, boxes, qids.input_ids.to(dev), mtok.convert_tokens_to_ids(TOR))
        cue = cue.unsqueeze(0).to(torch.bfloat16)
        kw = prepare_vlm_batch(proc, r, dev, add_answer=r["answer"])
        labels = answer_labels(proc, r, kw["input_ids"], dev)
        out = vlm(**kw, labels=labels, tor_embeds=cue, tor_token_id=tid)
        return out.loss

    model.train()
    import time
    t0 = time.time()
    for i, r in enumerate(rows[:args.steps]):
        loss = step(r)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if i % 10 == 0:
            print(f"{i}/{min(args.steps, len(rows))} loss={loss.item():.4f} "
                  f"grad={float(gn):.3f} {time.time() - t0:.0f}s", flush=True)
        if args.smoke and i + 1 >= args.smoke:
            break
    torch.save(model.state_dict(), args.save)
    print("saved", args.save)


if __name__ == "__main__":
    main()
