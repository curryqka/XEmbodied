# Evaluation Normalization Protocol

This document specifies how benchmark scores are normalized for cross-model comparison, addressing reviewer questions about why the same model can receive different average scores in different tables.

## Eq. (10): Per-Metric Normalization

For each evaluation metric, let `Y = [y_1, ..., y_n]` denote the raw results of all compared models. The normalized score is:

```
ŷ_i = (y_i + ε) / (max(Y) + ε)    if higher is better
ŷ_i = (min(Y) + ε) / (y_i + ε)    if lower is better (e.g., RMSE)
```

where ε is a small positive constant for numerical stability. The best-performing model receives a normalized score of 1. Normalized values are multiplied by 100 for reporting.

## Reference Set Y

**The choice of Y determines the normalized scale.** Different tables use different Y sets:

| Table | Reference set Y | Effect |
|-------|----------------|--------|
| Table 1 (main comparison) | All models in Table 1 (GPT-4o, Qwen variants, etc.) | Strong baselines → scores range ~35–84 |
| Table 3 (architecture ablation) | 6 rows in Table 3 | Same backbone → scores range ~56–90 |
| Table 12 (curriculum ablation) | 5 rows in Table 12 | Same backbone → scores range ~60–96 |

**The same model can receive different normalized scores in different tables because Y changes.** This is by design (Eq. 10 normalizes relative to the comparison pool), but it means scores from different tables are **not directly comparable**.

## Why Table 3 reports 78.45 and Table 12 reports 90.13

Both Table 3 (row f) and Table 12 (row e) describe the full SFT model, but:
- Table 3 normalizes against the Table 1 model pool → full model gets 78.45.
- Table 12 normalizes against only the 5 curriculum rows → full model gets 90.13.

The 11.68-point difference is entirely due to the reference set, not performance.

## Recommendation

For camera-ready: report one normalization reference set for all tables, or provide raw (unnormalized) per-metric scores so reviewers can recompute. The 5:6:7 category weighting (spatial:semantic:embodied) is applied on top of the per-metric normalization and is consistent across tables.
