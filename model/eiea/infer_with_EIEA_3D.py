#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-image EIEA + 3DA inference demo through MS-Swift.

The register file auto-injects a TOR cache at the 10 ``<|endoftext|>``
placeholder positions.  The cache must have the same width as the model's word
embeddings (2048 for Qwen3-VL-30B-A3B).  If no matching cache is supplied the
register warns and runs without physical-cue injection.

Example:
    python model/eiea/infer_with_EIEA_3D.py \
      --model_path /dev/shm/models/xembodied-v0 \
      --image demo/front.jpg \
      --tor_cache /path/to/cue_10x2048.pt
"""
import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
TOR_STR = "<|endoftext|>"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path",
                    default=os.environ.get("XEMB_MODEL_DIR",
                                           "/dev/shm/models/xembodied-v0"))
    ap.add_argument("--image", required=True)
    ap.add_argument("--tor_cache", default=os.environ.get("XEMB_TOR_CACHE", ""),
                    help="10 x d_vlm TOR cache; omit to run without cue injection")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--prompt", default=(
        "Task Description:\n"
        "Estimate the spatial relation between the objects in the image.\n\n"
        "Question:\n"
        "What is the spatial relation between the two vehicles?\n\n"
        "Finally, provide a concise and definitive response in the <answer> "
        "tag. Use the following format:\n<answer>[Final answer]</answer>"))
    return ap.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"model path does not exist: {args.model_path}")
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"image not found: {args.image}")

    if args.tor_cache:
        os.environ["XEMB_TOR_CACHE"] = args.tor_cache
    os.environ.setdefault(
        "VGGT_REPO_DIR",
        os.path.join(os.path.dirname(HERE), "3da"))

    from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
    from swift.utils import import_external_file

    register = os.path.join(HERE, "qwen3_vl_vggt_register.py")
    logger.info("registering %s", register)
    import_external_file(register)

    engine = TransformersEngine(
        model=args.model_path,
        model_type="qwen3_vl_moe_vggt",
        torch_dtype="bfloat16",
        trust_remote_code=True,
        device_map="auto",
    )
    # Ten placeholders are required by the TOR injection path.
    text = TOR_STR * 10 + "\n" + args.prompt
    req = InferRequest(messages=[{"role": "user", "content": f"<image>{text}"}],
                       images=[args.image])
    cfg = RequestConfig(max_tokens=args.max_tokens,
                        temperature=args.temperature)
    resp = engine.infer([req], cfg)[0]
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    sys.exit(main())
