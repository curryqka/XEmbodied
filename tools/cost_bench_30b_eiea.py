#!/usr/bin/env python3
"""30B EIEA inference cost: 10 cue tokens injected via the EIEA register."""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, "/home/kaneqian/XEmbodied/code/xembodied-code - AAAI/xembodied/EIEA/models")

OUT = "/home/kaneqian/XEmbodied/results/eiea_box"
TOR_STR = "<|endoftext|>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--max_tokens", type=int, default=32)
    ap.add_argument("--model_path", default="/dev/shm/models/xembodied-v0")
    args = ap.parse_args()
    from swift.utils import import_external_file
    from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
    import_external_file("/tmp/eiea_register_2048.py")
    eng = TransformersEngine(model=args.model_path, model_type="qwen3_vl_moe_vggt",
                             torch_dtype="bfloat16", trust_remote_code=True,
                             device_map="auto", max_memory={0: "43GiB", 1: "43GiB"})
    rows = json.load(open(f"{OUT}/val_cues.json"))[:args.limit]

    def req(r):
        p = (TOR_STR * 10 + "\n" + r["question"].replace("<image>", "")
             + "\nRespond with only the final answer in <answer></answer>; do not explain.")
        return InferRequest(messages=[{"role": "user", "content": "<image>" + p}],
                            images=[r["images"][1]])

    def timed(r, mt):
        torch.cuda.synchronize(); t = time.perf_counter()
        resp = eng.infer([req(r)], RequestConfig(max_tokens=mt, temperature=0,
                                                 stop=["</answer>"]))[0]
        torch.cuda.synchronize()
        return time.perf_counter() - t, resp

    for r in rows[:2]:
        timed(r, 8)
    rec = []
    for i, r in enumerate(rows):
        tp, _ = timed(r, 1)
        tt, resp = timed(r, args.max_tokens)
        u = getattr(resp, "usage", None)
        pt = (u.get("prompt_tokens") if isinstance(u, dict) else getattr(u, "prompt_tokens", None))
        rec.append({"prompt_tokens": pt, "prefill": tp, "total": tt, "decode": tt - tp})
        if (i + 1) % 10 == 0:
            print(i + 1, "/", len(rows), flush=True)
    out = {f: statistics.mean(x[f] for x in rec if x[f] is not None)
           for f in ["prompt_tokens", "prefill", "decode", "total"]}
    out["n"] = len(rec)
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{OUT}/cost_30b_eiea.json", "w"), indent=1)


if __name__ == "__main__":
    main()
