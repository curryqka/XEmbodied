#!/usr/bin/env python3
"""Evaluate the trained 30B EIEA on the same Swift path as the no-cue/text-cue baselines."""
import argparse
import json
import os
import re
import shutil
import sys
import time

import torch

sys.path.insert(0, "/home/kaneqian/XEmbodied/code/xembodied-code - AAAI/xembodied/EIEA/models")
OUT = "/home/kaneqian/XEmbodied/results/eiea_box"
CUE_DIR = "/tmp/eiea_cues"
CUR = "/tmp/eiea_cue_current.pt"
REG = "/tmp/eiea_register_swift.py"
TOR_STR = "<|endoftext|>"


def score(pred, gt, options):
    gu = str(gt).strip().upper(); up = pred.strip().upper()

    def norm(s):
        return re.sub(r"[^a-z0-9.]+", " ", s.lower()).strip()

    if re.fullmatch(r"[A-D]", gu):
        if re.fullmatch(r"[\(\[]?[A-D][\)\].:]?", up.strip()):
            return int(up.strip("().[]: ")[0] == gu)
        m = re.search(r"(?:answer is|answer\s*[:：])\s*[\(\[]?([A-D])\b", up)
        if m:
            return int(m.group(1) == gu)
        if options:
            p = norm(pred)
            for o in options:
                mm = re.match(r"\s*([A-D])[\.\):\s]*(.*)", o)
                if mm and norm(mm.group(2)) and norm(mm.group(2)) in p:
                    return int(mm.group(1) == gu)
        letters = re.findall(r"(?:^|[\s:：])([A-D])[\.\):：]", up)
        return int((letters[-1] if letters else up[:1]) == gu)
    try:
        g = float(gt)
    except ValueError:
        return int(gu in up)
    nums = re.findall(r"-?\d+(?:\.\d+)?", pred)
    return abs(float(nums[0]) - g) if nums else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--model_path", default="/dev/shm/models/xembodied-v0")
    ap.add_argument("--out", default=os.path.join(OUT, "eval_eiea_30b_swift.json"))
    ap.add_argument("--cpu_layers", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--cue_dir", default=CUE_DIR)
    ap.add_argument("--n_images", type=int, default=1)
    ap.add_argument("--timing", action="store_true")
    args = ap.parse_args()
    # register with the mutable cache path
    src = ("/home/kaneqian/XEmbodied/code/xembodied-code - AAAI/xembodied/EIEA/models/"
           "qwen3_vl_vggt_register.py")
    s = open(src).read()
    s = s.replace('TOR_EMBEDS_PATH = os.path.join(os.path.dirname(__file__), "tor_embeds_cache_fb.pt")',
                  f'TOR_EMBEDS_PATH = "{CUR}"')
    s = s.replace('_DEFAULT_VGGT_REPO_DIR = str(Path(__file__).resolve().parents[3] / "vggt")',
                  '_DEFAULT_VGGT_REPO_DIR = "/home/kaneqian/XEmbodied/code/xembodied-code - AAAI/xembodied/3DA/my_qwen3_vggt_xattnv2/vggt"')
    open(REG, "w").write(s)

    from swift.utils import import_external_file
    from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
    import_external_file(REG)
    if args.cpu_layers > 0:
        from transformers import AutoConfig
        n = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True).text_config.num_hidden_layers
        dm = {f"model.language_model.layers.{i}": ("cpu" if i < args.cpu_layers else 1)
              for i in range(n)}
        dm.update({"model.language_model.embed_tokens": 1, "model.language_model.norm": 1,
                   "model.language_model.rotary_emb": 1, "model.visual": 1,
                   "spatial_encoder": 1, "connector": 1, "lm_head": 1})
        eng = TransformersEngine(model=args.model_path, model_type="qwen3_vl_moe_vggt",
                                 torch_dtype="bfloat16", trust_remote_code=True,
                                 device_map=dm)
    else:
        eng = TransformersEngine(model=args.model_path, model_type="qwen3_vl_moe_vggt",
                                 torch_dtype="bfloat16", trust_remote_code=True,
                                 device_map="auto", max_memory={0: "43GiB", 1: "43GiB"})
    rows = json.load(open(os.path.join(OUT, "val_cues.json")))[:args.limit]
    recs = []
    for i, r in enumerate(rows):
        shutil.copyfile(os.path.join(args.cue_dir, f"{i}.pt"), CUR)
        p = (TOR_STR * 10 + "\n" + r["question"].replace("<image>", "")
             + "\nRespond with only the final answer in <answer></answer>; do not explain.")
        imgs = r["images"] if args.n_images >= 6 else [r["images"][1]]
        req = InferRequest(messages=[{"role": "user",
                                      "content": "".join("<image>" for _ in imgs) + p}],
                           images=imgs)
        resp = eng.infer([req], RequestConfig(max_tokens=args.max_tokens, temperature=0,
                                              stop=["</answer>"]))[0]
        raw = resp.choices[0].message.content
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, re.I | re.S)
        pred = (m.group(1) if m else raw).strip()
        rec = {"qid": f"q{i}", "gt": r["answer"], "pred": pred, "raw": raw,
               "ok": score(pred, r["answer"], r.get("options"))}
        if args.timing:
            from swift.infer_engine import RequestConfig as RC
            torch.cuda.synchronize(); t0 = time.perf_counter()
            eng.infer([req], RC(max_tokens=1, temperature=0, stop=["</answer>"]))
            torch.cuda.synchronize(); tp = time.perf_counter() - t0
            rec["prefill"] = tp
            u = getattr(resp, "usage", None)
            rec["prompt_tokens"] = (u.get("prompt_tokens") if isinstance(u, dict)
                                    else getattr(u, "prompt_tokens", None))
        recs.append(rec)
        if (i + 1) % 10 == 0:
            print(i + 1, "/", len(rows), flush=True)

    mc = [x for x in recs if isinstance(x["ok"], int) and not _num(x["gt"])]
    num = [x for x in recs if isinstance(x["ok"], float)]
    summary = {"n": len(recs), "mc_n": len(mc),
               "mc_acc": sum(x["ok"] for x in mc) / len(mc) if mc else None,
               "num_n": len(num),
               "num_mae": sum(abs(x["ok"]) for x in num) / len(num) if num else None}
    print(summary)
    json.dump({"summary": summary, "predictions": recs}, open(args.out, "w"), indent=1, ensure_ascii=False)
    with open(args.out.replace(".json", ".jsonl"), "w") as f:
        for x in recs:
            if not _num(x["gt"]):
                f.write(json.dumps({"qid": x["qid"], "correct": int(x["ok"])}) + "\n")
            else:
                f.write(json.dumps({"qid": x["qid"], "pred": float(x["gt"]) + abs(x["ok"]),
                                    "gt": float(x["gt"])}) + "\n")


def _num(x):
    try:
        float(x); return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
