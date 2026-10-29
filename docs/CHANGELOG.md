# Changelog

All notable fixes and additions made during code review are recorded here.

## [0.2.0] — 2026-10-29

### Fixed

- **`model/eiea/cue_encoder.py`** rewritten. The previous file was not
  runnable: it contained a dead placeholder slice, extracted TOR states by a
  brittle offset, called a non-existent `self.midi.tor_proj`, defined an
  unused cross-attention module, and placed the latent queries *before* the
  question text. The new encoder follows the paper supplement + rebuttal
  sequence (`[X_T; X_img; X_depth; Z_box; Z_BEVmap; (Z_BEVflow); R_1..R_K]`),
  takes the last `K` hidden states, and projects with
  `RMSNorm → Linear → GELU → Linear`. It handles empty/missing box, BEV, depth
  and flow cues, freezes the Mamba body by default, and supports
  checkpoint save/load.
- **`training/train_eiea_30b.py`** was a syntax-error stub
  (`sys.path.insert(0, PROJ_DIR = ...)`) that broke after one loop iteration.
  It is now a working frozen-30B trainer with box + BEV + depth cues, random
  modality masking, partial-cue dropping, box jitter and BEV/depth noise.
- **`training/train_eiea_30b_bev.py`** imported the private `train_e2e` /
  `prep` names and a non-existent BEV pipeline. It is now a thin deprecated
  alias for `train_eiea_30b.py`.
- **`training/train_eiea_box.py`** now trains the same encoder on the 7B SSR
  proxy with the same corruption recipe and a correct 3584-d cue width.
- **`evaluation/eval_eiea_30b.py`** no longer imports `train_e2e` /
  `E2EEncoder`; it loads `EIEACueEncoder` checkpoints and evaluates clean plus
  15 stress conditions (drop, partial, jitter, mask, noise, wrong-scene,
  no-cue), reporting paired deltas vs clean and a cue-injection sanity flag.
- **TOR injection (`model/eiea/qwen3_vl_moe_vggt.py`)** now validates the
  marker count and cue length, accepts an explicit `tor_positions` mask, handles
  `[K,D]`/`[B,K,D]` cues, and raises instead of silently skipping injection.
  Decode steps with no prompt markers are still skipped.
- **TOR register (`model/eiea/qwen3_vl_vggt_register.py`)** validates the
  cache width against the model's word-embedding size. The shipped 10×3584
  cache is skipped with a loud warning for the 30B (2048-d) model instead of
  crashing in `masked_scatter`; `XEMB_EIEA_STRICT_CACHE=1` turns this into an
  error.
- **VGGT default paths** in the model/register files now resolve to the
  in-repo `model/3da` copy instead of the private `.../3DA/my_qwen3_vggt_xattnv2`
  tree. The 3DA register also adds its own directory to `sys.path` so
  `from modeling_qwen3_vl_moe_vggt import ...` works in the flat repo layout.
- **`scripts/infer_demo.py`** defaults now point at `model/3da` and a
  `demo/sample.jpg` that is generated on first run (the README previously
  referenced a non-existent `./infer_demo.py` and image).
- **`docs/EIEA_ARCHITECTURE.md`** corrected the sequence order (latent queries
  are last, not before the text), the `tor_start` description (it omitted BEV
  tokens), and the 10-vs-64 / 2048-vs-8192 clarifications.

### Added

- **`model/eiea/corruption.py`**: random modality masking, partial-cue
  curriculum, box element drop/jitter, BEV mask/noise/channel drop, depth
  noise/mask/flatten, plus fixed evaluation conditions.
- **`model/eiea/preprocess.py`**: standalone VGGT 518×518 letterbox
  preprocessing (no Swift import).
- **`training/eiea_data.py`**: shared row loading, cue loading, prompt/label
  construction and scoring.
- **`docs/EIEA_TRAINING.md`**, **`model/eiea/README.md`**: recipe, presets,
  commands, environment variables and scope.
- **`model/eiea/infer_with_EIEA_3D.py`**: runnable EIEA inference consumer
  (the README previously pointed at a file that did not exist). It passes the
  10 TOR markers, resolves VGGT to the in-repo `model/3da`, and honors
  `XEMB_TOR_CACHE`.
- **`tests/test_eiea.py`**: 15 CPU tests for encoder shapes, missing cues,
  freeze behaviour / gradient reach, checkpoint round-trip, corruption
  invariants, all eval conditions, data helpers and scoring.

## [0.1.1] — 2026-10-28

### Fixed

- **3DA model (`model/3da/modeling_qwen3_vl_moe_vggt.py` L141–143)**:
  Removed dead `delattr` loop that would delete the entire `vggt_model`
  when any head parameter name matched. The loop was a safety net for
  cleanup after `del self.vggt_model.<head>` already handled removal;
  it was unreachable but misleading. Replaced with a commented safety-net
  that checks the top-level attribute name before deleting.

### Added

- **`model/eiea/cue_encoder.py`**: Complete EIEA cue encoder architecture:
  - `MIDIBackbone`: Mamba-2 sequential encoder with image/depth projections
  - `BoxEncoder`: 3D box → token encoder (class embedding + MLP)
  - `BEVEncoder`: BEV grid → token encoder (CNN → spatial pool → Linear)
  - `CueProjector`: d_mamba → d_VLM projection (RMSNorm → Linear → GELU → Linear)
  - `EIEACueEncoder`: full pipeline combining all sub-encoders

- **`model/eiea/__init__.py`**, **`model/__init__.py`**: Package init files.

- **`model/3da/vggt/__init__.py`**, `model/3da/vggt/models/__init__.py`,
  `model/3da/vggt/heads/__init__.py`, `model/3da/vggt/utils/__init__.py`,
  `model/3da/__init__.py`: Missing package init files.

- **`docs/EIEA_ARCHITECTURE.md`**: Forward pass specification addressing
  reviewer questions about Mamba input ordering, retained positions,
  token count (10 vs 64), and dimensional consistency (2048 vs 3584).

- **`docs/NORMALIZATION.md`**: Eq. (10) normalization protocol with
  explicit reference-set definition, explaining why Table 3 reports
  78.45 while Table 12 reports 90.13 for the same model.

- **`tools/paired_stats.py`**: Paired bootstrap CI + exact McNemar test
  for ablation comparison.

### Removed

- Dead code: commented-out `self.vggt_embedding_merger` alias and
  `self.post_init()` in `Qwen3VLMoeVGGTForConditionalGeneration.__init__`.
- Debug print statements and unused imports.

## [0.1.0] — 2026-10-25

### Added

- 3DA model: VGGT spatial encoder, embedding merger, Qwen3-VL-30B integration
- EIEA model: TOR injection interface
- 4-stage training scripts (sft_ddp_s1–s4)
- Dataset registration (dataset.py)
- Evaluation scripts for 18 benchmarks
- Data conversion tools for 12+ datasets
- Inference demo
