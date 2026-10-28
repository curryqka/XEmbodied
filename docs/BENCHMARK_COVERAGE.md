# Benchmark Coverage

The paper reports results on 18 public benchmarks across three task domains. The 5:6:7 weighting (spatial:semantic:embodied) reflects the relative dataset quantities.

## Domain → Benchmark → Evaluation Script Mapping

| # | Domain | Benchmark | Metric | Evaluation Script |
|---|--------|-----------|--------|-------------------|
| 1 | Spatial & 3D | Ego3D-Bench ACC | accuracy | `eval_ego3dbenchr1.py` |
| 2 | Spatial & 3D | Ego3D-Bench RMSE | distance (↓) | `eval_ego3dbenchr1.py` |
| 3 | Spatial & 3D | SURDS | accuracy | `eval_SURDSv2.py` |
| 4 | Spatial & 3D | VLADBench (spatial) | accuracy | `eval_vladbenchv2.py` |
| 5 | Spatial & 3D | STRIDE-QA | accuracy | `eval_SURDSv2.py` (shared) |
| 6 | Semantic | DriveBench | accuracy | `eval_drivebench_withqwenv3.py` |
| 7 | Semantic | DriveLMM-o1 | accuracy | `eval_DriveLMMo1v2.py` |
| 8 | Semantic | MapLM-v2 | accuracy | `eval_maplmv2.py` |
| 9 | Semantic | LingoQA | score | `eval_lingoQAv1.py` |
| 10 | Semantic | Omnidrive | accuracy | `eval_omnidrive.py` |
| 11 | Semantic | Cosmos-R1 | accuracy | `eval_cosmos.py` |
| 12 | Embodied | Affordance-2K | accuracy | `benchmark_sv2.py` |
| 13 | Embodied | Robo-Afford | accuracy | `benchmark_sv2.py` (shared) |
| 14 | Embodied | Cosmos-R1 (embodied) | accuracy | `eval_cosmos.py` (shared) |
| 15 | Embodied | Embodied-R1 | accuracy | `benchmark_sv2.py` (shared) |
| 16 | Embodied | RoboRefitBench | accuracy | `benchmark_sv2.py` (shared) |
| 17 | Embodied | VABench-Point | accuracy | `benchmark_sv2.py` (shared) |
| 18 | Embodied | Where2Place | accuracy | `benchmark_sv2.py` (shared) |

## Notes

- Ego3D-Bench counts as 2 metrics (ACC and RMSE).
- VLADBench is split into spatial (elements, traffic graph, object attributes) and semantic (traffic knowledge, ego decision) sub-scores.
- STRIDE-QA spans both spatial and semantic categories.
- Cosmos-R1 appears in both spatial (OOD) and embodied evaluations.
- The normalization reference set Y (Eq. 10 in supplement) differs per table; see `NORMALIZATION.md`.
