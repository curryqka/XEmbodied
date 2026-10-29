#!/usr/bin/env python3
"""Inference cost on XEmbodied 30B: no-cue vs text-cue prompt (front view)."""
import argparse
import json
import os
import statistics
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
OUT = os.environ.get("EIEABOX_OUT", os.path.join(ROOT, "results", "eiea_box"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--max_tokens", type=int, default=32)
    ap.add_argument("--model_path", default="/dev/shm/models/xembodied-v0")
    args = ap.parse_args()
    from swift.utils import import_external_file
    from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
    os.environ.setdefault("VGGT_REPO_DIR", os.path.join(ROOT, "model", "3da"))
    reg = os.path.join(ROOT, "model", "3da", "qwen3_vl_moe_vggt_register.py")
    import_external_file(reg)
    eng = TransformersEngine(model=args.model_path, model_type="qwen3_vl_moe_vggt",
                             torch_dtype="bfloat16", trust_remote_code=True,
                             device_map="auto", max_memory={0: "43GiB", 1: "43GiB"})
    rows = json.load(open(os.path.join(OUT, "val_cues.json")))[:args.limit]

    def req(r, cues):
        p = r["question"].replace("<image>", "")
        if cues:
            p += ("\n3D object boxes in ego frame (meters):\n" + r.get("cue_text", ""))
        p += "\nRespond with only the final answer in <answer></answer>; do not explain."
        return InferRequest(messages=[{"role": "user", "content": "<image>" + p}],
                            images=[r["images"][1]])

    def timed(r, cues, mt):
        torch.cuda.synchronize()
        t = time.perf_counter()
        resp = eng.infer([req(r, cues)], RequestConfig(max_tokens=mt, temperature=0,
                                                       stop=["</answer>"]))[0]
        torch.cuda.synchronize()
        return time.perf_counter() - t, resp

    for r in rows[:2]:
        timed(r, False, 8); timed(r, True, 8)
    res = {"no_cue": [], "text_cue": []}
    for i, r in enumerate(rows):
        for key, cues in [("no_cue", False), ("text_cue", True)]:
            tp, _ = timed(r, cues, 1)
            tt, resp = timed(r, cues, args.max_tokens)
            u = getattr(resp, "usage", None)
            pt = (u.get("prompt_tokens") if isinstance(u, dict) else getattr(u, "prompt_tokens", None))
            res[key].append({"prompt_tokens": pt, "prefill": tp, "total": tt,
                             "decode": tt - tp})
        if (i + 1) % 10 == 0:
            print(i + 1, "/", len(rows), flush=True)

    out = {}
    for k, rows_ in res.items():
        out[k] = {f: statistics.mean(x[f] for x in rows_ if x[f] is not None)
                  for f in ["prompt_tokens", "prefill", "decode", "total"]}
        out[k]["n"] = len(rows_)
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{OUT}/cost_30b.json", "w"), indent=1)


if __name__ == "__main__":
    main()
