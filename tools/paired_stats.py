#!/usr/bin/env python3
"""
Paired comparison of two runs on the SAME question list (rebuttal exps B/C/D).

Reports accuracy of each run, the difference (B - A) in points, the relative change
(B - A) / A, a paired-bootstrap 95% CI of the difference, and an exact McNemar p-value.
For numeric questions it reports MAE / RMSE of each run and the paired difference.

Input: two JSONL files, one row per question, same `qid`s:
  {"qid": "...", "correct": 0 or 1}                      # multiple choice
  {"qid": "...", "pred": 12.3, "gt": 10.0}               # numeric (metres)
Usage:
  python paired_stats.py --a runs/clean.jsonl --b runs/drop_all.jsonl --name drop_all
"""
import argparse, json, math, random


def load(path):
    return {r["qid"]: r for r in (json.loads(l) for l in open(path) if l.strip())}


def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference run, e.g. clean cues")
    ap.add_argument("--b", required=True, help="compared run")
    ap.add_argument("--name", default="B")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    qids = sorted(set(A) & set(B))
    if len(qids) < len(A) or len(qids) < len(B):
        print(f"warning: only {len(qids)} shared qids (A={len(A)}, B={len(B)})")
    rng = random.Random(args.seed)

    mc = [q for q in qids if "correct" in A[q]]
    if mc:
        a = [A[q]["correct"] for q in mc]; b = [B[q]["correct"] for q in mc]
        acc_a, acc_b = 100 * sum(a) / len(a), 100 * sum(b) / len(b)
        diffs = []
        for _ in range(args.boot):
            idx = [rng.randrange(len(mc)) for _ in mc]
            diffs.append(100 * sum(b[i] - a[i] for i in idx) / len(idx))
        diffs.sort()
        lo, hi = diffs[int(0.025 * args.boot)], diffs[int(0.975 * args.boot)]
        n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
        n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
        rel = (acc_b - acc_a) / acc_a * 100 if acc_a else float("nan")
        print(f"[MC] n={len(mc)}  A={acc_a:.2f}  {args.name}={acc_b:.2f}  "
              f"diff={acc_b - acc_a:+.2f} pts  rel={rel:+.1f}%  95% CI [{lo:+.2f}, {hi:+.2f}]  "
              f"McNemar p={mcnemar_exact(n10, n01):.3f}")

    num = [q for q in qids if "pred" in A[q]]
    if num:
        ea = [abs(A[q]["pred"] - A[q]["gt"]) for q in num]
        eb = [abs(B[q]["pred"] - B[q]["gt"]) for q in num]
        mae_a, mae_b = sum(ea) / len(ea), sum(eb) / len(eb)
        rmse_a = math.sqrt(sum(e * e for e in ea) / len(ea))
        rmse_b = math.sqrt(sum(e * e for e in eb) / len(eb))
        print(f"[NUM] n={len(num)}  MAE A={mae_a:.2f} {args.name}={mae_b:.2f}  "
              f"RMSE A={rmse_a:.2f} {args.name}={rmse_b:.2f}  "
              f"rel RMSE={(rmse_b - rmse_a) / rmse_a * 100:+.1f}%")


if __name__ == "__main__":
    main()