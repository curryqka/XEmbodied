# UPDATE: Replace placeholder paths with your actual paths.
import json
import os
import argparse
from typing import List, Tuple, Dict, Any
from pathlib import Path
from PIL import Image
import random

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

def points_to_bbox(points_list: List[List[List[float]]]) -> List[List[float]]:
    """
    Convert a list of point sets to bounding boxes.
    
    Args:
        points_list: List of point sets, where each set is a list of [x, y] points
        
    Returns:
        List of bounding boxes [[x1, y1, x2, y2], ...]
    """
    bboxes = []
    
    for point_set in points_list:
        if not point_set:
            continue
            
        # Flatten points
        all_points = []
        for point in point_set:
            if isinstance(point, list) and len(point) >= 2:
                all_points.append((point[0], point[1]))
        
        if not all_points:
            continue
            
        # Calculate bounding box
        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]
        
        x1 = int(min(x_coords))
        y1 = int(min(y_coords))
        x2 = int(max(x_coords))
        y2 = int(max(y_coords))
        
        bboxes.append([x1, y1, x2, y2])
    
    return bboxes

def process_roboafford_sample(sample: Dict[str, Any], 
                             image_dir: str,
                             coord_mode: str = 'thousand') -> Dict[str, Any]:
    """
    Process a single RoboAfford sample.
    
    Args:
        sample: Raw sample from RoboAfford dataset
        image_dir: Directory containing images
        coord_mode: Coordinate normalization mode
        
    Returns:
        Processed sample in ms-swift format
    """
    # Get image path and dimensions
    img_filename = sample["img"]
    img_path = os.path.join(image_dir, img_filename)
    
    # Get image dimensions
    width, height = 640, 480  # Default fallback
    try:
        with Image.open(img_path) as img:
            width, height = img.size
    except Exception as e:
        print(f"Warning: Could not read image {img_path}: {e}")
    
    # Process points to get bounding boxes
    points_data = sample.get("answer", [])
    bboxes_absolute = points_to_bbox(points_data)
    
    # Normalize coordinates
    bboxes_normalized = []
    for bbox in bboxes_absolute:
        normalized_bbox = normalize_coordinates(bbox, width, height, coord_mode)
        bboxes_normalized.append(normalized_bbox)
    
    # Construct user question
    original_question = sample["question"]
    
    # Modify the question to ask for bounding boxes instead of points
    if "list of tuples" in original_question and "coordinates" in original_question:
        # Replace point-based instruction with bbox-based instruction
        question = original_question.split("Your answer should be formatted as")[0]
        if coord_mode == 'thousand':
            question += "Your answer should be formatted as a list of bounding boxes, i.e. [[x1, y1, x2, y2], ...], where each bounding box contains the normalized coordinates (0-1000 scale) of the region that should be gripped. The coordinates should be integers representing the normalized pixel locations."
        elif coord_mode == 'normalized':
            question += "Your answer should be formatted as a list of bounding boxes, i.e. [[x1, y1, x2, y2], ...], where each bounding box contains the normalized coordinates (0-1 scale) of the region that should be gripped. The coordinates should be rounded to two decimal places."
        else:  # absolute
            question += "Your answer should be formatted as a list of bounding boxes, i.e. [[x1, y1, x2, y2], ...], where each bounding box contains the absolute pixel coordinates of the region that should be gripped. The coordinates should be integers."
    else:
        # If the pattern doesn't match, add bbox instruction
        if coord_mode == 'thousand':
            question = original_question + " Provide the answer as a list of bounding boxes in normalized coordinates (0-1000 scale) in the format: [[x1, y1, x2, y2], ...]"
        elif coord_mode == 'normalized':
            question = original_question + " Provide the answer as a list of bounding boxes in normalized coordinates (0-1 scale) in the format: [[x1, y1, x2, y2], ...]"
        else:  # absolute
            question = original_question + " Provide the answer as a list of bounding boxes in absolute coordinates in the format: [[x1, y1, x2, y2], ...]"
    
    # Construct assistant answer
    bbox_str = json.dumps(bboxes_normalized)
    assistant_answer = f"<answer>{bbox_str}</answer>"
    
    # Prepare metadata
    metadata = {
        "original_category": sample.get("category", ""),
        "original_img": sample["img"],
        "original_mask": sample.get("mask", ""),
        "image_width": width,
        "image_height": height,
        "coord_mode": coord_mode,
        "num_bboxes": len(bboxes_normalized)
    }
    
    # Construct ms-swift format
    ms_swift_sample = {
        "messages": [
            {
                "role": "system", 
                "content": "You are a helpful assistant that can understand images and provide precise spatial information. You will be shown images and asked about object affordances. Provide your answers in the specified format."
            },
            {
                "role": "user",
                "content": f"<image>\n{question}"
            },
            {
                "role": "assistant",
                "content": assistant_answer
            }
        ],
        "images": [img_path],
        "meta_data": metadata
    }
    
    return ms_swift_sample

def convert_roboafford_to_ms_swift(input_file: str, 
                                  output_dir: str, 
                                  convert_ratio: float = 1.0,
                                  coord_mode: str = 'thousand',
                                  image_base_dir: str = None):
    """
    Convert RoboAfford dataset to ms-swift format.
    
    Args:
        input_file: Path to input JSON file
        output_dir: Path to output directory
        convert_ratio: Ratio of data to convert (for debugging)
        coord_mode: Coordinate normalization mode
        image_base_dir: Base directory for images
    """
    # Set default image base directory if not provided
    if image_base_dir is None:
        # Use the parent directory of the input file
        input_dir = os.path.dirname(input_file)
        image_base_dir = os.path.join(input_dir, "images")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset name for output file
    dataset_name = "RoboAfford"
    output_file = os.path.join(output_dir, f"{dataset_name}_ms_swift.jsonl")
    
    # Load input data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples from {input_file}")
    
    # Apply conversion ratio
    if convert_ratio < 1.0:
        num_samples = int(len(data) * convert_ratio)
        data = random.sample(data, num_samples)
        print(f"Converting {num_samples} samples ({convert_ratio*100:.1f}%)")
    
    # Process each sample
    processed_samples = []
    skipped_samples = 0
    
    for i, sample in enumerate(data):
        try:
            ms_swift_sample = process_roboafford_sample(
                sample, 
                image_base_dir, 
                coord_mode
            )
            processed_samples.append(ms_swift_sample)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            skipped_samples += 1
            continue
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} samples...")
    
    # Save to JSONL file
    with open(output_file, 'w') as f:
        for sample in processed_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Successfully converted {len(processed_samples)} samples")
    print(f"Skipped {skipped_samples} samples")
    print(f"Saved to: {output_file}")
    
    # Also save a few examples for inspection
    examples_file = os.path.join(output_dir, f"{dataset_name}_examples.json")
    with open(examples_file, 'w') as f:
        json.dump(processed_samples[:5], f, indent=2, ensure_ascii=False)
    
    print(f"Example samples saved to: {examples_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert RoboAfford dataset to ms-swift format")
    
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="data/public_robo_datasets/RoboAfford-Eval/annotations_absxy.json",
        help="Path to input JSON file"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/preprocess_data/all_eval_data/RoboAfford",
        help="Path to output directory"
    )
    
    parser.add_argument(
        "--convert_ratio", 
        type=float, 
        default=1.0,
        help="Ratio of data to convert (for debugging)"
    )
    
    parser.add_argument(
        "--coord_mode", 
        type=str, 
        default="absolute",
        choices=["absolute", "thousand", "normalized"],
        help="Coordinate normalization mode"
    )
    
    parser.add_argument(
        "--image_base_dir", 
        type=str, 
        default="data/public_robo_datasets/RoboAfford-Eval/images",
        help="Base directory containing images"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("Configuration:")
    print(f"  Input file: {args.input_file}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Convert ratio: {args.convert_ratio}")
    print(f"  Coordinate mode: {args.coord_mode}")
    print(f"  Image base directory: {args.image_base_dir}")
    print()
    
    # Run conversion
    convert_roboafford_to_ms_swift(
        input_file=args.input_file,
        output_dir=args.output_dir,
        convert_ratio=args.convert_ratio,
        coord_mode=args.coord_mode,
        image_base_dir=args.image_base_dir
    )

if __name__ == "__main__":
    main()