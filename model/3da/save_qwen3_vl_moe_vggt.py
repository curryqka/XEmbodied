"""
Save the full fused weight shards for Qwen3-VL-30B-A3B + VGGT.

Usage:
- Load base Qwen3-VL weights + fused model architecture
- Load VGGT pretrained weights and write to corresponding prefix in the fused model
- Save as a complete loadable HF checkpoint (config + safetensors shards + tokenizer/processor files)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _parse_torch_dtype(s: str) -> torch.dtype:
    s = str(s).lower().strip()
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16", "half"}:
        return torch.float16
    if s in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {s}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge VGGT into Qwen3-VL and save FULL fused weights.")
    parser.add_argument("--base_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vggt_ckpt_path", required=True)
    parser.add_argument("--vggt_repo_dir", required=True)

    parser.add_argument("--vggt_image_size", type=int, default=518)
    parser.add_argument("--vggt_patch_size", type=int, default=14)
    parser.add_argument("--vggt_embed_dim", type=int, default=1024)
    parser.add_argument("--spatial_embeds_layer_idx", type=int, default=-1)

    # NOTE: `device_map="auto"` may trigger accelerate tied-weights edge cases on some versions.
    # For saving a fused checkpoint, we do not need automatic sharding. Default to single-device load.
    # Use a concrete device string to avoid ambiguity (e.g. "cuda" has no index).
    parser.add_argument("--device_map", default="cuda:0")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--max_shard_size", default="4GB")
    parser.add_argument("--safe_serialization", type=int, default=1)
    parser.add_argument("--vggt_prefix_in_model", default="spatial_encoder.vggt_model.")
    args = parser.parse_args()

    from accelerate import Accelerator  # pyright: ignore[reportMissingImports]
    from transformers import AutoConfig  # pyright: ignore[reportMissingImports]

    base_dir = Path(args.base_model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import local fused model class.
    # This file lives at: .../my_qwen3_vggt_xattn_mlp/plugin/save_*.py
    # Project root is its parent dir.
    project_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_dir))

    # Import vggt.
    sys.path.insert(0, str(Path(args.vggt_repo_dir).resolve()))

    from modeling_qwen3_vl_moe_vggt import Qwen3VLMoeVGGTForConditionalGeneration  # noqa: WPS433
    from vggt.models.vggt import VGGT  # noqa: WPS433

    # Load fused model (Qwen3 weights).
    config = AutoConfig.from_pretrained(str(base_dir), trust_remote_code=True)
    config.architectures = ["Qwen3VLMoeVGGTForConditionalGeneration"]
    # IMPORTANT:
    # - Do NOT persist absolute local paths into config.json, otherwise multi-node/DDP runs may fail.
    # - VGGT weights are merged into the fused checkpoint, so `vggt_ckpt_path` must not be needed at load time.
    # - The VGGT code path should be provided by your runtime environment (e.g. PYTHONPATH / VGGT_REPO_DIR env).
    for _k in ("vggt_ckpt_path", "vggt_repo_dir"):
        if hasattr(config, _k):
            delattr(config, _k)
    config.vggt_spatial_config = {
        "img_size": int(args.vggt_image_size),
        "patch_size": int(args.vggt_patch_size),
        "embed_dim": int(args.vggt_embed_dim),
        "spatial_embeds_layer_idx": int(args.spatial_embeds_layer_idx),
    }
    config.vggt_connector_config = {"spatial_embeds_layer_idx": int(args.spatial_embeds_layer_idx)}

    model = Qwen3VLMoeVGGTForConditionalGeneration.from_pretrained(
        str(base_dir),
        device_map=args.device_map,
        dtype=_parse_torch_dtype(args.torch_dtype),
        config=config,
    ).eval()

    # Load vggt.
    vggt = VGGT(
        img_size=int(args.vggt_image_size),
        patch_size=int(args.vggt_patch_size),
        embed_dim=int(args.vggt_embed_dim),
    ).eval()
    sd = torch.load(str(args.vggt_ckpt_path), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if isinstance(sd, dict):
        cleaned = {}
        for k, v in sd.items():
            if isinstance(k, str) and k.startswith("module."):
                k = k[len("module.") :]
            cleaned[k] = v
        sd = cleaned
    vggt.load_state_dict(sd, strict=False)

    # Merge weights.
    print("Merging VGGT weights into Qwen3-VL model...")
    merge_count = 0
    fused_sd = model.state_dict()
    prefix = str(args.vggt_prefix_in_model)
    for name, tensor in vggt.state_dict().items():
        name_full = prefix + name
        if name_full in fused_sd:
            dst = fused_sd[name_full]
            fused_sd[name_full].copy_(tensor.to(device=dst.device, dtype=dst.dtype))
            merge_count += 1
    print(f"Weight merging complete, {merge_count} parameters/buffers merged")

    # Save full fused model.
    model.config.save_pretrained(str(output_dir))
    accelerator = Accelerator()
    accelerator.save_model(
        model=model,
        save_directory=str(output_dir),
        max_shard_size=str(args.max_shard_size),
        safe_serialization=bool(args.safe_serialization),
    )

    # Copy tokenizer / processor files so the fused dir is a complete HF checkpoint.
    import shutil

    _aux_patterns = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "chat_template.json",
        "generation_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ]
    copied = []
    for name in _aux_patterns:
        src = base_dir / name
        if src.exists():
            shutil.copy2(str(src), str(output_dir / name))
            copied.append(name)
    if copied:
        print(f"Copied auxiliary files from base_model_dir: {copied}")

    print("Model merging and saving completed successfully")


if __name__ == "__main__":
    main()

