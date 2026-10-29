#!/usr/bin/env python3
"""Evaluate the trained 30B EIEA path (SSR cue encoder + 3584->2048 projection)."""
import argparse
import importlib.util
import json
import os
import re
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ssr_eiea_common import TOR, load_midi, load_visual_encoders
from ssr.utils.prompt import insert_tor
from train_e2e import E2EEncoder

MODEL_DIR = "/dev/shm/models/xembodied-v0"
EIEA_DIR = "/home/kaneqian/XEmbodied/code/xembodied-code - AAAI/xembodied/EIEA/models"
PROJ = ("/home/kaneqian/XEmbodied/code/xembodied-code - AAAI/xembodied/3DA/"
        "my_qwen3_vggt_xattnv2")
OUT = "/home/kaneqian/XEmbodied/results/eiea_box"
TOR_ID = 151643
TOR_STR = "<|endoftext|>"
TERSE = "\nRespond with only the final answer in <answer></answer>; do not explain."
CONDS = ["clean", "no_boxes", "wrong_scene", "no_cue"]


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
    ap.add_argument("--ckpt", default=os.path.join(OUT, "eiea_30b.pt"))
    ap.add_argument("--out", default=os.path.join(OUT, "eval_eiea_30b.json"))
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

    cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    n = cfg.text_config.num_hidden_layers
    dm = {f"model.language_model.layers.{i}": (0 if i < 22 else 1) for i in range(n)}
    dm.update({"model.language_model.embed_tokens": 0, "model.language_model.norm": 1,
               "model.language_model.rotary_emb": 0, "model.visual": 0,
               "spatial_encoder": 0, "connector": 0, "lm_head": 1})
    cfg.vggt_device = "cuda:0"; cfg.vggt_dtype = "bfloat16"
    model = C.from_pretrained(MODEL_DIR, config=cfg, dtype=torch.bfloat16,
                              device_map=dm, trust_remote_code=True)
    model.eval()
    proc = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    # match the 512-image-token budget used in the cost benchmarks
    try:
        proc.image_processor.max_pixels = 512 * 16 * 16 * 4
        proc.image_processor.min_pixels = 4 * 16 * 16 * 4
    except Exception as e:
        print("pixel cap not set:", e, flush=True)
    tps = int(cfg.vision_config.temporal_patch_size)
    tok = proc.tokenizer
    spec2 = importlib.util.spec_from_file_location(
        "xreg3da", PROJ + "/plugin/qwen3_vl_moe_vggt_register.py")
    reg = importlib.util.module_from_spec(spec2); sys.modules["xreg3da"] = reg
    spec2.loader.exec_module(reg)
    prep = reg._preprocess_image_to_vggt_tensor

    ck = torch.load(args.ckpt, map_location="cpu")
    midi, mtok = load_midi("cuda:1"); midi.load_state_dict(ck["midi"])
    encoder = E2EEncoder(midi, use_boxes=True, use_bev=False).to("cuda:1")
    encoder.load_state_dict({"midi": ck["midi"], **ck["enc"]}, strict=False)
    encoder.eval()
    proj = nn.Linear(3584, 2048).to("cuda:1")
    proj.load_state_dict(ck["proj"]); proj.eval()
    cp, clip, sp, sig = load_visual_encoders("cuda:1")
    tor_mtok = mtok.convert_tokens_to_ids(TOR)

    def cue_for(r, boxes):
        img = Image.open(r["images"][1]).convert("RGB"); dep = np.load(r["depth"])
        with torch.no_grad():
            ie = clip(**cp(images=img, return_tensors="pt").to("cuda:1")).last_hidden_state.float()
            de = sig(**sp(images=dep, return_tensors="pt").to("cuda:1")).last_hidden_state.float()
            ids = mtok(r["question"] + insert_tor("", 10), add_special_tokens=False, return_tensors="pt")
            bt = torch.tensor(boxes, dtype=torch.float32, device="cuda:1") if boxes else torch.zeros(0, 11, device="cuda:1")
            c = encoder(ie, de, bt, None, ids.input_ids.to("cuda:1"), tor_mtok)
            c = proj(c.float()).unsqueeze(0).to(torch.bfloat16)
        return c

    def generate(r, cue, mt, timing=None):
        text = (TOR_STR * 10 if cue is not None else "") + "\n" + r["question"] + TERSE
        img = Image.open(r["images"][1]).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": text}]}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs)
        enc = proc(text=[prompt], images=[img], return_tensors="pt")
        kw = {k: v.to("cuda:0") for k, v in enc.items()
              if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
        tchw = torch.stack([prep(img)] * (int(enc["image_grid_thw"][0, 0]) * tps), 0).to("cuda:0")
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            if cue is not None:
                o = model.generate(**kw, image_tchw=[tchw], max_new_tokens=mt,
                                   do_sample=False, stop_strings=["</answer>"],
                                   tokenizer=tok, tor_embeds=cue, tor_token_id=TOR_ID)
            else:
                o = model.generate(**kw, image_tchw=[tchw], max_new_tokens=mt,
                                   do_sample=False, stop_strings=["</answer>"], tokenizer=tok)
        torch.cuda.synchronize()
        return time.perf_counter() - t0, o, kw

    res = {c: [] for c in CONDS}
    rec = []
    for i, r in enumerate(rows):
        variants = {"clean": r["boxes"], "no_boxes": [],
                    "wrong_scene": rows[(i + 1) % len(rows)]["boxes"]}
        for c, bx in variants.items():
            cue = cue_for(r, bx)
            _, o, kw = generate(r, cue, 32)
            raw = tok.decode(o[0][kw["input_ids"].shape[1]:], skip_special_tokens=True)
            m = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, re.I | re.S)
            pred = (m.group(1) if m else raw).strip()
            res[c].append({"qid": f"q{i}", "gt": r["answer"], "pred": pred,
                           "ok": score(pred, r["answer"]), "n_tok": int(kw["input_ids"].shape[1])})
            if c == "clean":
                tp, _, _ = generate(r, cue, 1)
                tt, _, _ = generate(r, cue, 32)
                rec.append({"prompt_tokens": int(kw["input_ids"].shape[1]),
                            "prefill": tp, "total": tt, "decode": tt - tp})
        _, o, kw = generate(r, None, 32)
        raw = tok.decode(o[0][kw["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, re.I | re.S)
        pred = (m.group(1) if m else raw).strip()
        res["no_cue"].append({"qid": f"q{i}", "gt": r["answer"], "pred": pred,
                              "ok": score(pred, r["answer"])})
        if (i + 1) % 10 == 0:
            print(i + 1, "/", len(rows), flush=True)

    summary = {}
    for c, recs in res.items():
        mc = [x for x in recs if not _num(x["gt"])]
        vals = []
        for x in recs:
            if _num(x["gt"]):
                nums = re.findall(r"-?\d+(?:\.\d+)?", x["pred"])
                if nums:
                    vals.append(abs(float(nums[0]) - float(x["gt"])))
        summary[c] = {"n": len(recs), "mc_n": len(mc),
                      "mc_acc": statistics.mean(x["ok"] for x in mc) if mc else None,
                      "num_n": len(vals),
                      "num_mae": statistics.mean(vals) if vals else None}
    summary["cost"] = {f: statistics.mean(x[f] for x in rec) for f in
                       ["prompt_tokens", "prefill", "decode", "total"]}
    print(json.dumps(summary, indent=1))
    json.dump({"summary": summary, "predictions": res},
              open(args.out, "w"), indent=1, ensure_ascii=False)


def _num(x):
    try:
        float(x); return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
