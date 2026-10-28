# Changelog

All notable fixes and additions made during code review are recorded here.

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
