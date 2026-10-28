# Data Provenance and Cue Sources

This document clarifies the provenance of physical cues and data splits, addressing reviewer questions about OOD input sources and data contamination.

## Cue Sources

All physical cues used in reported results are constructed from **dataset ground-truth annotations**, not from online tool predictions. The encoders (DiffMap, PriorMotion, SG-Net) process these annotations into compact representations; they are not perception models run at inference time.

| Cue type | Source | Training split |
|----------|--------|---------------|
| 3D boxes | nuScenes / Argoverse / Waymo ground-truth annotations | Dataset official train split |
| BEV maps | Dataset-provided HD map rasters or ego-centric projections | Same |
| Motion / flow | Ground-truth object velocities from annotations | Same |

The encoders (DiffMap, PriorMotion, SG-Net) are trained on these annotations offline; at inference time, the pre-computed cue embeddings are cached and injected. No perception tool is run online during evaluation.

## Ego3D-Bench Scene Overlap

We verified that the nuScenes portion of Ego3D-Bench has no scene-level overlap with the nuScenes train split:

- Ego3D-Bench contains 1628 images (684 nuScenes, 944 Argoverse/Waymo).
- The 684 nuScenes images span 114 unique scenes, **all from the official val split**.
- Zero scenes overlap with the nuScenes train split used for training.

This confirms that the model's performance on Ego3D-Bench is not attributable to scene-level memorisation. Per-metric raw results for the curriculum ablation rows are not separately released; we recommend reporting raw per-metric values in the camera-ready version.

## Training / Evaluation Split

Training and evaluation use **disjoint scene sets**. The Ego3D-Bench evaluation set is the official benchmark split; no Ego3D-Bench questions, images, or scenes are used for training. Similarly, DriveBench, Cosmos-R1, and other OOD benchmarks are excluded from all training stages.

## Data Decontamination

We did not perform perceptual-hash or near-duplicate detection across training and evaluation sets, as the scene-level split already guarantees disjoint geographic sequences. For camera-ready, we plan to add image-level perceptual-hash statistics.
