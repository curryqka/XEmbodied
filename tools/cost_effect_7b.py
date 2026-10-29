#!/usr/bin/env python3
"""Cost + accuracy of three cue interfaces on the same 7B SSR-VLM:
no cue, text-serialized cue, EIEA 10-token injection."""
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.eiea.ssr_eiea_common import (
    TOR, VLM_BASE, load_midi, load_visual_encoders, load_vlm,
)
from ssr.utils.prompt import insert_tor

OUT = os.environ.get("EIEABOX_OUT", str(ROOT / "results" / "eiea_box"))


class BoxAdapter(nn.Module):
    """Cross-attention box adapter used by the rebuttal EIEA-lite (7B).

    Copied here so this cost script is self-contained; the training version
    lives in the private experiment tree.
    """

    def __init__(self, dim=3584, heads=8):
        super().__init__()
        self.class_emb = nn.Embedding(11, 16)
        self.box_mlp = nn.Sequential(
            nn.Linear(16 + 10, 512), nn.GELU(), nn.Linear(512, dim))
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, cue, boxes):
        cue_n = cue / (cue.norm() + 1e-6)
        if boxes.shape[0] > 0:
            tok = self.box_mlp(torch.cat([self.class_emb(boxes[:, 0].long()),
                                          boxes[:, 1:11]], -1)).unsqueeze(0)
            att = self.cross(cue_n.unsqueeze(0), tok, tok)[0].squeeze(0)
            att = att / (att.norm() + 1e-6)
        else:
            att = torch.zeros_like(cue_n)
        return cue_n + self.alpha * att
TERSE = "\nRespond with only the final answer in <answer></answer>; do not explain."
CONDS = ["no_cue", "text_cue", "eiea"]


def score(pred, gt):
    gu = str(gt).strip().upper(); up = pred.strip().upper()
    if re.fullmatch(r"[A-D]", gu):
        if re.fullmatch(r"[\(\[]?[A-D][\)\].:]?", up.strip()):
            return int(up.strip("().[]: ")[0] == gu)
        m = re.search(r"(?:answer is|answer\s*[:：])\s*[\(\[]?([A-D])\b", up)
        if m:
            return int(m.group(1) == gu)
        letters = re.findall(r"(?:^|[\s:：])([A-D])[\.\):：]", up)
        return int((letters[-1] if letters else up[:1]) == gu)
    try:
        g = float(gt)
    except ValueError:
        return int(gu in up)
    nums = re.findall(r"-?\d+(?:\.\d+)?", pred)
    return int(bool(nums) and abs(float(nums[0]) - g) < 1e-3)


def numeric_mae(recs):
    vals = []
    for r in recs:
        try:
            g = float(r["gt"])
        except ValueError:
            continue
        nums = re.findall(r"-?\d+(?:\.\d+)?", r["pred"])
        if nums:
            vals.append(abs(float(nums[0]) - g))
    return (statistics.mean(vals), len(vals)) if vals else (None, 0)


def main():
    rows = json.load(open(os.path.join(OUT, "val_cues.json")))[:100]
    dev = "cuda:1"; torch.cuda.set_device(1)
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    vlm = load_vlm(dev); vlm.eval()
    proc = AutoProcessor.from_pretrained(VLM_BASE)
    proc.tokenizer.add_tokens(TOR, special_tokens=True)
    tid = proc.tokenizer.convert_tokens_to_ids(TOR)
    midi, mtok = load_midi(dev); midi.eval()
    cp, clip, sp, sig = load_visual_encoders(dev)
    adapter = BoxAdapter().to(dev)
    adapter.load_state_dict(torch.load(os.path.join(OUT, "adapter.pt"), map_location=dev))
    adapter.eval()

    cues, cue_times = [], []
    for i, r in enumerate(rows):
        im = Image.open(r["images"][1]).convert("RGB")
        dep = np.load(r["depth"])
        t = time.perf_counter()
        with torch.no_grad():
            ie = clip(**cp(images=im, return_tensors="pt").to(dev)).last_hidden_state.float()
            de = sig(**sp(images=dep, return_tensors="pt").to(dev)).last_hidden_state.float()
            ids = mtok(r["question"] + insert_tor("", 10), add_special_tokens=False, return_tensors="pt")
            b = torch.tensor(r["boxes"], dtype=torch.float32, device=dev) if r["boxes"] else torch.zeros(0, 11, device=dev)
            # midi cue then adapter
            mask = torch.cat([torch.ones(1, ie.size(1) + de.size(1), dtype=torch.long), ids.attention_mask], 1)
            cue = midi(mamba_input_ids=ids.input_ids.to(dev), mamba_attention_mask=mask.to(dev),
                       image_embeds=ie, depth_embeds=de,
                       tor_token_id=(mtok.convert_tokens_to_ids(TOR), 0), alignment=False).tor_embeds
            cue = adapter(cue.squeeze(0), b)
        torch.cuda.synchronize()
        cue_times.append(time.perf_counter() - t)
        cues.append(cue.float().cpu())

    def build(r, cond):
        q = r["question"]
        if cond == "text_cue":
            q = q + "\n3D object boxes in ego frame (meters):\n" + r.get("cue_text", "")
        text = (insert_tor("", 10) if cond == "eiea" else "") + "\n" + q + TERSE
        content = [{"type": "image", "image": Image.open(r["images"][1]).convert("RGB").resize((256, 256))}]
        content.append({"type": "text", "text": text})
        msgs = [{"role": "user", "content": content}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs)
        return proc(text=[prompt], images=imgs, videos=None, padding=True, return_tensors="pt")

    def timed(inp, cue, mt):
        kw = {k: v.to(dev) for k, v in inp.items()
              if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.no_grad():
            if cue is not None:
                o = vlm.generate(**kw, max_new_tokens=mt, do_sample=False,
                                 tor_embeds=cue.unsqueeze(0).to(dev), tor_token_id=tid)
            else:
                o = vlm.generate(**kw, max_new_tokens=mt, do_sample=False)
        torch.cuda.synchronize()
        return time.perf_counter() - t, o, kw

    out = {c: [] for c in CONDS}
    for i, r in enumerate(rows):
        for cond in CONDS:
            inp = build(r, cond)
            tp, _, _ = timed(inp, cues[i] if cond == "eiea" else None, 1)
            tt, o, kw = timed(inp, cues[i] if cond == "eiea" else None, 32)
            raw = proc.tokenizer.decode(o[0][kw["input_ids"].shape[1]:], skip_special_tokens=True)
            m = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, re.I | re.S)
            pred = (m.group(1) if m else raw).strip()
            out[cond].append({"qid": f"q{i}", "gt": r["answer"], "pred": pred,
                              "ok": score(pred, r["answer"]),
                              "prompt_tokens": int(inp["input_ids"].shape[1]),
                              "prefill": tp, "total": tt, "decode": tt - tp})
        if (i + 1) % 10 == 0:
            print(i + 1, "/", len(rows), flush=True)

    summary = {"cue_gen_s": statistics.mean(cue_times)}
    for c, recs in out.items():
        mc = [x for x in recs if not _is_num(x["gt"])]
        mae, nnum = numeric_mae(recs)
        summary[c] = {"n": len(recs), "mc_n": len(mc),
                      "mc_acc": statistics.mean(x["ok"] for x in mc) if mc else None,
                      "num_mae": mae, "num_n": nnum,
                      "prompt_tokens": statistics.mean(x["prompt_tokens"] for x in recs),
                      "prefill": statistics.mean(x["prefill"] for x in recs),
                      "decode": statistics.mean(x["decode"] for x in recs),
                      "total": statistics.mean(x["total"] for x in recs)}
    print(json.dumps(summary, indent=1))
    json.dump({"summary": summary, "predictions": out},
              open(os.path.join(OUT, "cost_effect_7b.json"), "w"), indent=1, ensure_ascii=False)


def _is_num(x):
    try:
        float(x); return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
