#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL inference script.
Supports: native Qwen3-VL model + custom qwen3_vl_moe_vggt model.
"""
import os
import logging

# -------------------------- Environment Configuration --------------------------
# Uncomment and modify as needed:
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['MAX_PIXELS'] = '2080000'

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------------- Model Switch Configuration --------------------------
# Model mode: True = custom VGGT-fused model | False = native Qwen3-VL-MoE
USE_CUSTOM_MODEL = True

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Custom model configuration (your fine-tuned merged checkpoint)
CUSTOM_MODEL_CFG = {
    "model_path": os.environ.get(
        "XEMB_MODEL_DIR", os.path.join(_ROOT, "model", "3da", "base_ckpt")),
    "model_type": "qwen3_vl_moe_vggt",
    "register_file": os.path.join(
        _ROOT, "model", "3da", "qwen3_vl_moe_vggt_register.py"),
}

# Native Qwen3-VL model configuration
NATIVE_MODEL_CFG = {
    # <-- UPDATE: set to your Qwen3-VL-30B-A3B-Instruct path -->
    "model_path": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "model_type": "qwen3_vl_moe",
}

# Inference configuration
INFER_CFG = {
    "max_tokens": 512,
    "temperature": 0,
    "user_prompt": (
        "Task Description:\n"
        "The primary goal of this task is to estimate the vertical distance of the specified object "
        "in the image from the camera, which is positioned at the origin. You need to analyze the image "
        "and choose the correct range of distance from the camera based on the visual cues provided.\n\n"
        "Question:\n"
        "How far is the vertical distance of the white car in the picture from the camera?\n\n"
        "Finally, provide a concise and definitive response in the <answer> tag. Use the following format:\n"
        "<answer>[Final answer]</answer>"
    ),
    "image_paths": [os.environ.get(
        "XEMB_DEMO_IMAGE", os.path.join(_ROOT, "demo", "sample.jpg"))],
}


def _ensure_sample_image(path: str) -> None:
    """Create a synthetic quick-start image if the user did not supply one."""
    if os.path.isfile(path):
        return
    from PIL import Image, ImageDraw
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new("RGB", (640, 360), (165, 190, 215))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 250, 640, 360], fill=(90, 95, 100))       # road
    d.rectangle([250, 160, 390, 260], fill=(210, 210, 215))   # car
    d.ellipse([275, 245, 305, 270], fill=(30, 30, 30))
    d.ellipse([335, 245, 365, 270], fill=(30, 30, 30))
    img.save(path, quality=90)

# -------------------------- Dependency Import --------------------------
from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
from swift.utils import import_external_file

# Register custom model architecture if needed
if USE_CUSTOM_MODEL:
    logger.info(f"Loading custom model register file: {CUSTOM_MODEL_CFG['register_file']}")
    import_external_file(CUSTOM_MODEL_CFG["register_file"])

# -------------------------- Inference Engine Initialization --------------------------
def init_engine() -> TransformersEngine:
    """Initialize the inference engine (auto-adapts to native/custom model)."""
    model_cfg = CUSTOM_MODEL_CFG if USE_CUSTOM_MODEL else NATIVE_MODEL_CFG
    model_path = model_cfg["model_path"]
    model_type = model_cfg["model_type"]

    if not os.path.isdir(model_path) and not model_path.startswith("Qwen/"):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    logger.info(f"Initializing model: {model_type} | path: {model_path}")
    engine = TransformersEngine(
        model=model_path,
        model_type=model_type,
        torch_dtype="bfloat16",
        trust_remote_code=True,
    )
    return engine

# -------------------------- Core Inference Logic --------------------------
def infer_image(engine: TransformersEngine, image_path: str, prompt: str, config: RequestConfig):
    """Run inference on a single image."""
    if not os.path.isfile(image_path):
        logger.error(f"Image not found: {image_path}")
        return None

    infer_requests = [
        InferRequest(
            messages=[{"role": "user", "content": f"<image>{prompt}"}],
            images=[image_path],
        )
    ]

    resp_list = engine.infer(infer_requests, config)
    answer = resp_list[0].choices[0].message.content
    return answer

# -------------------------- Main --------------------------
if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Qwen3-VL inference task started")
    logger.info("=" * 50)

    for _p in INFER_CFG["image_paths"]:
        _ensure_sample_image(_p)

    # 1. Initialize engine
    infer_engine = init_engine()

    # 2. Initialize inference config
    req_config = RequestConfig(
        max_tokens=INFER_CFG["max_tokens"],
        temperature=INFER_CFG["temperature"],
    )

    # 3. Run inference on each image
    logger.info("Starting image inference...")
    for idx, img_path in enumerate(INFER_CFG["image_paths"]):
        logger.info(f"\n=== Inferring image {idx+1} ===")
        result = infer_image(
            engine=infer_engine,
            image_path=img_path,
            prompt=INFER_CFG["user_prompt"],
            config=req_config,
        )
        print(f"\nPrompt: {INFER_CFG['user_prompt']}")
        print(f"Model response: {result}")

    logger.info("\n" + "=" * 50)
    logger.info("Inference task completed!")
    logger.info("=" * 50)
