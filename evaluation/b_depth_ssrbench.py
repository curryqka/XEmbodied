#!/usr/bin/env python3
"""Improved B: depth-noise stress test on SSRBench (in-domain for SSR-VLM).

Same 69-question COCO subset as E2; corrupt the depth that feeds MIDI, regenerate
the 10 cue tokens, run SSR-VLM. Variants: clean, noise_20, mask_50, flat,
wrong_scene.
"""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.eiea.ssr_eiea_common import (
    DEPTH_CKPT, TOR, VLM_BASE, depth_uint8, load_midi, load_visual_encoders,
    load_vlm,
)
from training.eiea_data import extract_answer, score_answer

DATA = os.environ.get("SSR_DEPTH_DATA", "/media/nas_users/kaneqian/data/XEmbodied")
MANIFEST = os.environ.get("SSR_DEPTH_MANIFEST",
                          os.path.join(DATA, "ssrbench_coco_manifest.json"))
IMAGES = os.environ.get("SSR_DEPTH_IMAGES",
                        os.path.join(DATA, "ssrbench_images"))
OUT = os.environ.get("SSR_DEPTH_OUT", str(ROOT / "results" / "plan1025"))
VARIANTS = ["clean", "noise_20", "mask_50", "flat", "wrong_scene"]


def corrupt(depth, variant, rng, donor=None):
    d = depth.copy()
    r = np.random.default_rng(rng.randrange(2**31))
    span = float(d.max() - d.min()) or 1.0
    if variant == "clean":
        return d
    if variant == "noise_20":
        return np.clip(d.astype(np.float32) + r.normal(0, 0.20 * span, d.shape), 0, 255).astype(np.uint8)
    if variant == "mask_50":
        h, w = d.shape[:2]
        fill = int(d.mean())
        idx = [(i, j) for i in range(h // 8) for j in range(w // 8)]
        for i, j in rng.sample(idx, int(0.5 * len(idx))):
            d[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = fill
        return d
    if variant == "flat":
        return np.full_like(d, int(d.mean()))
    if variant == "wrong_scene":
        return donor.copy()
    raise ValueError(variant)


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.cuda.set_device(1)
    rows = json.load(open(MANIFEST))
    have = {f for f in os.listdir(IMAGES) if f.endswith(".jpg")}
    rows = [r for r in rows if r["image_path"].rsplit("/", 1)[1] in have]
    per = {}
    for r in rows:
        per.setdefault(r["task"], []).append(r)
    rows = []
    for t, rs in per.items():
        if t == "action recognition":
            continue
        rows.extend(rs[:16])
    print(len(rows), "SSRBench questions")

    from depth_pro import create_model_and_transforms
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    midi, mtok = load_midi("cuda:1")
    enc = load_visual_encoders("cuda:1")
    dpro, dtf = create_model_and_transforms(checkpoint_uri=DEPTH_CKPT,
                                           device=torch.device("cuda:1"))
    dpro.eval()
    tor_id = mtok.convert_tokens_to_ids(TOR)
    cp, clip, sp, sig = enc

    rng = random.Random(0)
    cleans, clipse = [], []
    for i, r in enumerate(rows):
        im = Image.open(os.path.join(IMAGES, r["image_path"].rsplit("/", 1)[1])).convert("RGB")
        cleans.append(depth_uint8(im, dpro, dtf, "cuda:1"))
        with torch.no_grad():
            clipse.append(clip(**cp(images=im, return_tensors="pt").to("cuda:1")).last_hidden_state.float())
        if (i + 1) % 20 == 0:
            print("depth", i + 1, "/", len(rows), flush=True)
    del dpro
    torch.cuda.empty_cache()

    cues = {v: [] for v in VARIANTS}
    for i, r in enumerate(rows):
        donor = cleans[(i + 1) % len(rows)]
        for v in VARIANTS:
            dep = corrupt(cleans[i], v, rng, donor)
            with torch.no_grad():
                de = sig(**sp(images=dep, return_tensors="pt").to("cuda:1")).last_hidden_state.float()
                ids = mtok(r["question"] + insert_tor("", 10), add_special_tokens=False, return_tensors="pt")
                mask = torch.cat([torch.ones(1, clipse[i].size(1) + de.size(1), dtype=torch.long),
                                  ids.attention_mask], 1)
                tor = midi(mamba_input_ids=ids.input_ids.to("cuda:1"),
                           mamba_attention_mask=mask.to("cuda:1"),
                           image_embeds=clipse[i], depth_embeds=de,
                           tor_token_id=(tor_id, 0), alignment=False).tor_embeds.squeeze(0).cpu()
            cues[v].append(tor)
        if (i + 1) % 10 == 0:
            print("cues", i + 1, "/", len(rows), flush=True)
    del midi, enc
    torch.cuda.empty_cache()

    vlm = load_vlm("cuda:1")
    proc = AutoProcessor.from_pretrained(VLM_BASE)
    proc.tokenizer.add_tokens(TOR, special_tokens=True)
    tid = proc.tokenizer.convert_tokens_to_ids(TOR)

    inputs = []
    for r in rows:
        im = Image.open(os.path.join(IMAGES, r["image_path"].rsplit("/", 1)[1])).convert("RGB").resize((256, 256))
        text = insert_tor("", 10) + "\n" + r["question"]
        msgs = [{"role": "user", "content": [{"type": "image", "image": im},
                                             {"type": "text", "text": text}]}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs)
        inputs.append(proc(text=[prompt], images=imgs, videos=None, padding=True, return_tensors="pt"))

    results = {v: [] for v in VARIANTS}
    for v in VARIANTS:
        for i, (r, inp) in enumerate(zip(rows, inputs)):
            kw = {k: t.to("cuda:1") for k, t in inp.items()
                  if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
            with torch.no_grad():
                out = vlm.generate(**kw, max_new_tokens=32, do_sample=False,
                                   tor_embeds=cues[v][i].unsqueeze(0).to("cuda:1"),
                                   tor_token_id=tid)
            raw = proc.tokenizer.decode(out[0][kw["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_answer(raw)
            results[v].append({"qid": f"q{i}", "task": r["task"], "gt": r["answer"],
                               "pred": pred, "raw": raw,
                               "ok": int(score_answer(pred, r["answer"]))})
        acc = np.mean([p["ok"] for p in results[v]])
        changed = sum(a["raw"] != b["raw"] for a, b in zip(results["clean"], results[v]))
        print(f"{v:12s} acc={acc:.4f} changed={changed}/{len(rows)}", flush=True)

    summary = {}
    for v in VARIANTS:
        recs = results[v]
        summary[v] = {"n": len(recs), "acc": float(np.mean([p["ok"] for p in recs])),
                      "changed_vs_clean": int(sum(a["raw"] != b["raw"]
                                                  for a, b in zip(results["clean"], recs)))}
        with open(os.path.join(OUT, f"Bs_{v}.jsonl"), "w") as f:
            for p in recs:
                f.write(json.dumps({"qid": p["qid"], "correct": p["ok"]}) + "\n")
    json.dump({"summary": summary, "predictions": results},
              open(os.path.join(OUT, "Bs_depth_results.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
