# UPDATE: Replace placeholder paths with your actual paths.
import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image
import numpy as np


def normalize_coordinates(bbox: List[int], width: int, height: int, 
                         coord_mode: str) -> List[float]:
    """
    Normalize coordinates according to the specified mode.
    
    Args:
        bbox: [x1, y1, x2, y2] in absolute coordinates
        width: image width
        height: image height
        coord_mode: normalization mode, one of ['absolute', 'thousand', 'normalized']
    
    Returns:
        List of box coordinates [x1, y1, x2, y2] in the specified coordinate system
    """
    x1, y1, x2, y2 = bbox
    
    if coord_mode == 'absolute':
        # Use absolute coordinates directly
        return [float(x1), float(y1), float(x2), float(y2)]
    
    elif coord_mode == 'thousand':
        # Convert to thousand-level normalized coordinates (0-1000)
        if width and height:
            x1_norm = int(x1 / width * 1000)
            y1_norm = int(y1 / height * 1000)
            x2_norm = int(x2 / width * 1000)
            y2_norm = int(y2 / height * 1000)
            return [float(x1_norm), float(y1_norm), float(x2_norm), float(y2_norm)]
        else:
            # Fallback to absolute if image dimensions not available
            return [float(x1), float(y1), float(x2), float(y2)]
    
    elif coord_mode == 'normalized':
        # Convert to normalized coordinates (0-1) and round to 2 decimal places
        if width and height:
            x1_norm = round(x1 / width, 2)
            y1_norm = round(y1 / height, 2)
            x2_norm = round(x2 / width, 2)
            y2_norm = round(y2 / height, 2)
            return [x1_norm, y1_norm, x2_norm, y2_norm]
        else:
            # Fallback to absolute if image dimensions not available
            return [float(x1), float(y1), float(x2), float(y2)]
    
    else:
        raise ValueError(f"Unknown coord_mode: {coord_mode}")


def get_mask_bounding_box(mask_path: str) -> Tuple[List[int], int, int]:
    """
    Get bounding box from mask image (white region is target).
    
    Args:
        mask_path: Path to mask image
    
    Returns:
        Tuple of (bounding_box, width, height)
        bounding_box: [x1, y1, x2, y2] in absolute coordinates
    """
    try:
        mask = Image.open(mask_path)
        width, height = mask.size
        
        # Convert mask to numpy array
        mask_array = np.array(mask)
        
        # Find white regions (value 255 for grayscale masks)
        if len(mask_array.shape) == 2:  # Grayscale mask
            white_pixels = mask_array == 255
        elif len(mask_array.shape) == 3:  # RGB mask
            # Check if all channels are white
            white_pixels = np.all(mask_array == 255, axis=2)
        else:
            raise ValueError(f"Unexpected mask shape: {mask_array.shape}")
        
        if not np.any(white_pixels):
            # No white pixels found, return full image
            return [0, 0, width, height], width, height
        
        # Find coordinates of white pixels
        rows, cols = np.where(white_pixels)
        
        if len(rows) == 0 or len(cols) == 0:
            return [0, 0, width, height], width, height
        
        x1 = int(np.min(cols))
        y1 = int(np.min(rows))
        x2 = int(np.max(cols)) + 1  # +1 because bounding box is exclusive
        y2 = int(np.max(rows)) + 1
        
        # Ensure bounding box is within image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        return [x1, y1, x2, y2], width, height
        
    except Exception as e:
        print(f"Error processing mask {mask_path}: {e}")
        return [0, 0, 100, 100], 100, 100  # Return a default small box


def process_part_affordance_data(
    input_file: str,
    output_dir: str,
    convert_ratio: float = 1.0,
    coord_mode: str = 'thousand',
    dataset_root: str = None
) -> None:
    """
    Process Part-Affordance-2K dataset and convert to target format.
    
    Args:
        input_file: Path to input metadata.json file
        output_dir: Output directory for converted data
        convert_ratio: Ratio of data to convert (for debugging)
        coord_mode: Coordinate normalization mode
        dataset_root: Root directory of dataset (if None, inferred from input_file)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine dataset root directory
    if dataset_root is None:
        # Infer from input file path: remove '/metadata/metadata.json' to get root
        dataset_root = os.path.dirname(os.path.dirname(os.path.abspath(input_file)))
    
    print(f"Dataset root: {dataset_root}")
    print(f"Input file: {input_file}")
    print(f"Output dir: {output_dir}")
    
    # Load input data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Apply convert ratio
    if convert_ratio < 1.0:
        num_samples = int(len(data) * convert_ratio)
        data = data[:num_samples]
        print(f"Converting {num_samples} samples ({convert_ratio*100:.1f}%)")
    else:
        print(f"Converting all {len(data)} samples")
    
    # Process each sample
    converted_data = []
    skipped_count = 0
    
    for i, sample in enumerate(data):
        if i % 100 == 0:
            print(f"Processing sample {i}/{len(data)}...")
        
        # Get paths
        question_id = sample.get("question_id", i)
        problem = sample.get("problem", "")
        category_type = sample.get("category_type", "")
        image_path_rel = sample.get("image_path", "")
        mask_path_rel = sample.get("mask_path", "")
        
        # Construct absolute paths
        image_path = os.path.join(dataset_root, image_path_rel)
        mask_path = os.path.join(dataset_root, mask_path_rel)
        
        # Check if files exist
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            skipped_count += 1
            continue
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found: {mask_path}")
            skipped_count += 1
            continue
        
        # Get bounding box from mask
        bbox_abs, img_width, img_height = get_mask_bounding_box(mask_path)
        
        # Normalize coordinates
        bbox_norm = normalize_coordinates(bbox_abs, img_width, img_height, coord_mode)
        
        # Format bounding box string
        bbox_str = f"[{bbox_norm[0]:.2f}, {bbox_norm[1]:.2f}, {bbox_norm[2]:.2f}, {bbox_norm[3]:.2f}]"
        
        # Prepare system prompt
        system_prompt = "You are a helpful visual assistant that can locate objects in images. Provide the bounding box of the target region in response to the user's request."
        
        # Prepare user content
        # Capitalize first letter and add period if needed
        if problem and not problem.endswith('.'):
            problem = problem + '.'
        
        if problem and problem[0].islower():
            problem = problem[0].upper() + problem[1:]
        
        # user_content = f"<image>\n{problem} Please provide the bounding box of the target region in the image. Your answer should be in the format: <answer>[x1, y1, x2, y2]</answer> where coordinates are normalized."
        user_content = f"<image>\n{problem} To execute the manipulation, please provide the bounding box of the target region in the image. Your answer should be in the format: <answer>[x1, y1, x2, y2]</answer>."

        # Prepare assistant content
        assistant_content = f"<answer>{bbox_str}</answer>"
        
        # Prepare meta_data
        meta_data = {
            "question_id": question_id,
            "category_type": category_type,
            "mask_path": mask_path_rel,
            "original_problem": problem,
            "image_dimensions": {"width": img_width, "height": img_height},
            "absolute_bbox": bbox_abs,
            "normalized_bbox": bbox_norm,
            "coord_mode": coord_mode
        }
        
        # Create converted sample
        converted_sample = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ],
            "images": [image_path],
            "meta_data": meta_data
        }
        
        converted_data.append(converted_sample)
    
    print(f"\nProcessing complete!")
    print(f"Successfully converted: {len(converted_data)} samples")
    print(f"Skipped: {skipped_count} samples")
    
    # Save converted data
    output_file = os.path.join(output_dir, "part_affordance_ms_swift.jsonl")
    
    with open(output_file, 'w') as f:
        for sample in converted_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\nConverted data saved to: {output_file}")
    
    # Also save a small sample for verification
    if len(converted_data) > 0:
        sample_file = os.path.join(output_dir, "sample_converted.json")
        with open(sample_file, 'w') as f:
            json.dump(converted_data[0], f, indent=2, ensure_ascii=False)
        print(f"Sample saved to: {sample_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert Part-Affordance-2K dataset to ms-swift format")
    
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/preprocess_data/Part-Affordance-2K/metadata/metadata.json",
        help="Path to input metadata.json file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/preprocess_data/all_eval_data/Part-Affordance-2K/",
        help="Output directory for converted data"
    )
    
    parser.add_argument(
        "--convert_ratio",
        type=float,
        default=1.0,
        help="Ratio of data to convert (0.0-1.0, for debugging)"
    )
    
    parser.add_argument(
        "--coord_mode",
        type=str,
        default="absolute",
        choices=["absolute", "thousand", "normalized"],
        help="Coordinate normalization mode"
    )
    
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Root directory of dataset (if None, inferred from input_file)"
    )
    
    args = parser.parse_args()
    
    # Process the data
    process_part_affordance_data(
        input_file=args.input_file,
        output_dir=args.output_dir,
        convert_ratio=args.convert_ratio,
        coord_mode=args.coord_mode,
        dataset_root=args.dataset_root
    )


if __name__ == "__main__":
    main()