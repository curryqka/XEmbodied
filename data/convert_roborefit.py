# UPDATE: Replace placeholder paths with your actual paths.
import json
import os, re
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import argparse, random

def parse_coordinate_string(coord_str: str) -> List[Tuple[float, float]]:
    """
    Parse coordinate string, e.g., "(339,234),(403,299)"
    Returns: [(339, 234), (403, 299)]
    """
    pattern = r'\((\d+),(\d+)\)'
    matches = re.findall(pattern, coord_str)
    coords = [(float(x), float(y)) for x, y in matches]
    return coords


def format_coordinate_string(coords: List[Tuple[float, float]], tag: str) -> str:
    """
    Format coordinate list as string
    """
    coord_str = ",".join([f"{x:.3f}" for coord in coords for x in coord])
    return f"<{tag}>[{coord_str}]</{tag}>"


# def normalize_coordinates_with_image_size(content: str, width: int, height: int) -> str:
#     """
#     Normalize box coordinates based on image size
#     """
#     box_pattern = r'<box>\((.*?)\)</box>'
#     def normalize_box(match):
#         box_content = match.group(1)
#         coords = parse_coordinate_string(f"({box_content})")
#         normalized_coords = [(x * 1000.0 / width, y * 1000.0 / height) for x, y in coords]
#         # Return formatted string
#         return format_coordinate_string(normalized_coords, "answer")

#     content = re.sub(box_pattern, normalize_box, content)
#     return content

def normalize_coordinates_with_image_size(content: str, width: int, height: int) -> str:
    """
    Normalize box coordinates based on image size
    """
    # Extract box coordinates from content
    box_pattern = r'<box>\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)</box>'
    match = re.search(box_pattern, content)

    if not match:
        # If no box coordinates matched, return original content
        return content

    # Correctly extract four coordinates
    groups = match.groups()
    if len(groups) != 4:
        return content

    x1, y1, x2, y2 = map(int, groups)

    # Normalize coordinates
    x1_norm = x1 * 1000.0 / width
    y1_norm = y1 * 1000.0 / height
    x2_norm = x2 * 1000.0 / width
    y2_norm = y2 * 1000.0 / height

    # Format output
    return f'<answer>[{x1_norm:.0f}, {y1_norm:.0f}, {x2_norm:.0f}, {y2_norm:.0f}]</answer>'


def convert_roborefit_sample(metadata_item: Dict[str, Any],
                           base_dir: str,
                           normalize_coords: bool = False) -> Optional[Dict[str, Any]]:
    sample_id = metadata_item.get("id", "")
    image_path = metadata_item["image_path"]
    conversation_path = os.path.join(base_dir, metadata_item["conversation_path"])

    if not os.path.exists(conversation_path):
        print(f"Warning: Conversation file not found: {conversation_path}")
        return None

    with open(conversation_path, 'r', encoding='utf-8') as f:
        conversation_data = json.load(f)

    full_image_path = os.path.join(base_dir, image_path.lstrip('/'))

    if not os.path.exists(full_image_path):
        print(f"Warning: Image file not found: {full_image_path}")
        return None

    width, height = 1000, 1000  # Default values
    if normalize_coords:
        try:
            with Image.open(full_image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Warning: Failed to get image size for {full_image_path}: {e}")
            # Continue with default values

    messages = []
    system_prompt = "You are a helpful assistant. I need you to identify objects in the image based on the instruction."
    messages.append({"role": "system", "content": system_prompt})

    for turn in conversation_data:
        role = turn.get("role", "")
        content = turn.get("content", "")
        # breakpoint()

        if not content or role not in ["user", "assistant"]:
            continue

        if role == "user":
            original_instruction = content.replace("<image>", "").strip()
            target_obj = original_instruction.split('the')[1]
            enhanced_instruction = (
                f"<image>\n"
                f"Please carefully analyze the image and provide the coordinates for the object mentioned in the instruction.\n"
                f"Instruction: {original_instruction}\n"
                f"The target object is:{target_obj}\n"
                f"Response format: Provide the bounding box of the target object in <answer>[x1, y1, x2, y2]</answer> format."
            )

            messages.append({"role": "user", "content": enhanced_instruction})

        elif role == "assistant":
            if normalize_coords:
                try:
                    content = normalize_coordinates_with_image_size(content, width, height)
                    # content = content.split('<points>')[0]
                    # breakpoint()
                except Exception as e:
                    print(f"Warning: Failed to normalize coordinates for {sample_id}: {e}")
                    # Keep original coordinates
            # "content": "<box>(339,234),(403,299)</box><points>(268,370),(370,267)</points>"
            # Define regex to match coordinates inside box tag
            # content_box_pattern = r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'

            # Search for match
            # match = re.search(content_box_pattern, content)

            # if match:
            #     # Extract four coordinate numbers and convert to integers
            #     coordinates = [int(match.group(i)) for i in range(1, 5)]

            # x1, y1, x2, y2 = coordinates
            # content = f'<answer>[{x1}, {y1}, {x2}, {y2}]</answer>'
            messages.append({"role": "assistant", "content": content})

    output_data = {
        "messages": messages,
        "images": [full_image_path]
    }

    output_data["meta_data"] = {
        "original_id": sample_id,
        "has_image": metadata_item.get("has_image", False),
        "has_conversation": metadata_item.get("has_conversation", False),
        "source_file": metadata_item.get("source_file", ""),
        "file_index": metadata_item.get("file_index", 0),
        "row_index": metadata_item.get("row_index", 0),
        "image_width": width if normalize_coords else None,
        "image_height": height if normalize_coords else None
    }

    return output_data


def main():
    parser = argparse.ArgumentParser(description="Convert roborefit-sft data to ms-swift format")
    parser.add_argument("--input_file", type=str, required=True,
                       help="Path to metadata.json file")
    parser.add_argument("--output_dir", type=str,
                       default="data/preprocess_data/roborefit-sft",
                       help="Output directory")
    parser.add_argument("--convert_ratio", type=float, default=1.0,
                       help="Ratio of data to convert (for debugging)")
    parser.add_argument("--normalize_coords", action="store_true",
                       help="Normalize coordinates using image size (x/1000 * width, y/1000 * height)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling")

    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Reading metadata from: {args.input_file}")
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_name = "roborefit-sft"
    output_file = os.path.join(args.output_dir, f"{dataset_name}_converted.jsonl")

    if args.convert_ratio < 1.0:
        num_samples = int(len(metadata) * args.convert_ratio)
        selected_indices = random.sample(range(len(metadata)), num_samples)
        samples_to_process = [metadata[i] for i in selected_indices]
        print(f"Sampling {num_samples} samples (ratio: {args.convert_ratio})")
    else:
        samples_to_process = metadata

    base_dir = os.path.dirname(args.input_file)

    converted_count = 0
    error_count = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i, metadata_item in enumerate(samples_to_process):
            if i % 1000 == 0 and i > 0:
                print(f"Processed {i} samples...")

            try:
                converted_sample = convert_roborefit_sample(
                    metadata_item,
                    base_dir,
                    args.normalize_coords
                )

                if converted_sample:
                    f_out.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                    converted_count += 1
                else:
                    error_count += 1

            except Exception as e:
                print(f"Error converting sample {metadata_item.get('id', f'index_{i}')}: {e}")
                error_count += 1

    stats = {
        "total_samples": len(samples_to_process),
        "converted_count": converted_count,
        "error_count": error_count,
        "convert_ratio": args.convert_ratio,
        "normalize_coords": args.normalize_coords,
        "output_file": output_file
    }

    stats_file = os.path.join(args.output_dir, f"{dataset_name}_conversion_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\nConversion Complete!")
    print(f"Total samples to process: {len(samples_to_process)}")
    print(f"Successfully converted: {converted_count}")
    print(f"Errors: {error_count}")
    print(f"Output saved to: {output_file}")
    print(f"Stats saved to: {stats_file}")

if __name__ == "__main__":
    main()
