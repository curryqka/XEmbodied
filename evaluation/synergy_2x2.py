#!/usr/bin/env python3
"""2x2 factorial: 3DA on/off x EIEA cue on/off on the same 100 questions.
3DA-off is implemented by replacing the spatial encoder's output with zeros
(same connector path, no geometric evidence)."""
import argparse
import importlib.util
import json
import os
import re
import statistics
import sys

import torch
from PIL import Image

MODEL_DIR = "/dev/shm/models/xembodied-v0"
EIEA_DIR = "/home/kaneqian/XEmbodied/code/xembodied-code - AAAI/xembodied/EIEA/models"
PROJ = ("/home/kaneqian/XEmbodied/code/xembodied-code - AAAI/xembodied/3DA/"
        "my_qwen3_vggt_xattnv2")
OUT = "/home/kaneqian/XEmbodied/results/eiea_box"
CUE_DIR = "/tmp/eiea_cues"
TOR_ID = 151643
TOR_STR = "<|endoftext|>"
TERSE = "\nRespond with only the final answer in <answer></answer>; do not explain."


def zero_forward(patch_start_idx=5):
    @torch.no_grad()
    def forward(image_tchw):
        tensors = image_tchw if isinstance(image_tchw, list) else [image_tchw]
        outs, idxs = [], []
        for t in tensors:
            p = (t.shape[-2] // 14) * (t.shape[-1] // 14)
            tok = torch.zeros(t.shape[0], patch_start_idx + p, 2048,
                              dtype=t.dtype, device=t.device)
            outs.append([tok])
            idxs.append(patch_start_idx)
        return outs, idxs
    return forward


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(OUT, "synergy_2x2.json"))
    args = ap.parse_args()
    rows = json.load(open(os.path.join(OUT, "val_cues.json")))[:args.limit]
    sys.path.insert(0, PROJ); sys.path.insert(0, EIEA_DIR)
    from transformers import AutoConfig, AutoProcessor
    from qwen_vl_utils import process_vision_info
    spec = importlib.util.spec_from_file_location(
        "eiea_modeling", os.path.join(EIEA_DIR, "qwen3_vl_moe_vggt.py"))
    mod = importlib.util.module_from_spec(spec); sys.modules["eiea_modeling"] = mod
    spec.loader.exec_module(mod)
    C = mod.Qwen3VLMoeVGGTForConditionalGeneration
    spec2 = importlib.util.spec_from_file_location(
        "xreg3da", PROJ + "/plugin/qwen3_vl_moe_vggt_register.py")
    reg = importlib.util.module_from_spec(spec2); sys.modules["xreg3da"] = reg
    spec2.loader.exec_module(reg)
    prep = reg._preprocess_image_to_vggt_tensor

    cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    n = cfg.text_config.num_hidden_layers
    dm = {f"model.language_model.layers.{i}": (0 if i < 24 else 1) for i in range(n)}
    dm.update({"model.language_model.embed_tokens": 0, "model.language_model.norm": 1,
               "model.language_model.rotary_emb": 0, "model.visual": 0,
               "spatial_encoder": 0, "connector": 0, "lm_head": 1})
    cfg.vggt_device = "cuda:0"; cfg.vggt_dtype = "bfloat16"
    model = C.from_pretrained(MODEL_DIR, config=cfg, dtype=torch.bfloat16,
                              device_map=dm, trust_remote_code=True)
    model.eval()
    proc = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    try:
        proc.image_processor.max_pixels = 512 * 16 * 16 * 4
        proc.image_processor.min_pixels = 4 * 16 * 16 * 4
    except Exception as e:
        print("pixel cap not set:", e, flush=True)
    tps = int(cfg.vision_config.temporal_patch_size)
    tok = proc.tokenizer
    orig_fwd = model.spatial_encoder.forward
    psi = int(model.spatial_encoder.vggt_model.aggregator.patch_start_idx)

    def generate(r, cue, three_da):
        model.spatial_encoder.forward = orig_fwd if three_da else zero_forward(psi)
        text = (TOR_STR * 10 if cue is not None else "") + "\n" + r["question"] + TERSE
        imgs = r["images"]
        content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in imgs]
        content.append({"type": "text", "text": text})
        msgs = [{"role": "user", "content": content}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        images, _ = process_vision_info(msgs)
        enc = proc(text=[prompt], images=images, return_tensors="pt")
        kw = {k: v.to("cuda:0") for k, v in enc.items()
              if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
        frames = []
        for p in imgs:
            frames.extend([prep(p)] * (int(enc["image_grid_thw"][0, 0]) * tps))
        tchw = torch.stack(frames, 0).to("cuda:0")
        with torch.no_grad():
            if cue is not None:
                o = model.generate(**kw, image_tchw=[tchw], max_new_tokens=16,
                                   do_sample=False, stop_strings=["</answer>"],
                                   tokenizer=tok, tor_embeds=cue, tor_token_id=TOR_ID)
            else:
                o = model.generate(**kw, image_tchw=[tchw], max_new_tokens=16,
                                   do_sample=False, stop_strings=["</answer>"], tokenizer=tok)
        raw = tok.decode(o[0][kw["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, re.I | re.S)
        return (m.group(1) if m else raw).strip()

    recs = {k: [] for k in ["3da_cue", "3da_nocue", "noda_cue", "noda_nocue"]}
    for i, r in enumerate(rows):
        cue = torch.load(os.path.join(CUE_DIR, f"{i}.pt"), map_location="cuda:0")["tor_embeds"]
        for key, c, d in [("3da_cue", cue, True), ("3da_nocue", None, True),
                          ("noda_cue", cue, False), ("noda_nocue", None, False)]:
            pred = generate(r, c, d)
            recs[key].append({"qid": f"q{i}", "gt": r["answer"], "pred": pred,
                              "ok": score(pred, r["answer"])})
        if (i + 1) % 10 == 0:
            print(i + 1, "/", len(rows), flush=True)

    summary = {}
    for k, rs in recs.items():
        mc = [x for x in rs if not _num(x["gt"])]
        summary[k] = {"mc_acc": statistics.mean(x["ok"] for x in mc), "n": len(mc)}
    a = summary["3da_cue"]["mc_acc"]; b = summary["3da_nocue"]["mc_acc"]
    c = summary["noda_cue"]["mc_acc"]; d = summary["noda_nocue"]["mc_acc"]
    summary["gain_3da"] = a - b
    summary["gain_noda"] = c - d
    summary["interaction"] = (a - b) - (c - d)
    print(json.dumps(summary, indent=1))
    json.dump({"summary": summary, "predictions": recs}, open(args.out, "w"),
              indent=1, ensure_ascii=False)


def _num(x):
    try:
        float(x); return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
