# UPDATE: Replace placeholder paths with your actual paths.
#!/usr/bin/env python3
import os
import json
import random
import argparse
from pathlib import Path

def convert_cosmos_to_msswift(input_file, output_dir, convert_ratio=1.0, video_root=None):
    """
    Convert Cosmos-R1-SFT JSON file to ms-swift format JSONL file.

    Args:
        input_file (str): Input JSON file path
        output_dir (str): Output directory
        convert_ratio (float): Conversion ratio, 0.0-1.0
        video_root (str, optional): Video file root directory, for building absolute paths
    """
    # Read input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Ensure data is a list
    if not isinstance(data, list):
        data = [data]

    # Sample by ratio
    if convert_ratio < 1.0:
        num_samples = int(len(data) * convert_ratio)
        data = random.sample(data, num_samples)
        print(f"Randomly sampled {num_samples}/{len(data)} samples")

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename
    input_filename = Path(input_file).stem
    output_file = os.path.join(output_dir, f"{input_filename}.jsonl")

    # Statistics
    converted_count = 0
    error_count = 0

    with open(output_file, 'w', encoding='utf-8') as fout:
        for sample in data:
            try:
                # 1. Get video path
                video_rel_path = sample.get("video", "")
                if not video_rel_path:
                    print(f"Warning: sample missing video field, skipping")
                    continue

                # Process video path
                if video_root:
                    # If video_root is provided, build absolute path
                    video_path = os.path.join(video_root, video_rel_path)
                else:
                    # Otherwise keep relative path
                    video_path = video_rel_path

                # 2. Process conversation messages
                conversations = sample.get("conversations", [])
                if not conversations:
                    print(f"Warning: sample missing conversations, skipping")
                    continue

                messages = []
                for msg in conversations:
                    role = msg.get("role", "")
                    content = msg.get("content", "")

                    # If user message, add <video> tag at the beginning
                    if role == "user":
                        # Add video tag, separated by newline
                        content = f"<video>\n{content}"

                    messages.append({
                        "role": role,
                        "content": content
                    })

                # 3. Build output object
                output_obj = {
                    "messages": messages,
                    "videos": [video_path]
                }

                # 4. Add metadata
                meta_data = sample.get("metadata", {})
                if meta_data:
                    output_obj["meta_data"] = meta_data

                # Write JSONL
                fout.write(json.dumps(output_obj, ensure_ascii=False) + "\n")
                converted_count += 1

            except Exception as e:
                print(f"Error processing sample: {e}")
                error_count += 1
                continue

    print(f"Conversion complete!")
    print(f"Success: {converted_count}, Failed: {error_count}")
    print(f"Output file: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert Cosmos-R1-SFT dataset to ms-swift format")
    parser.add_argument("--input_file", type=str, required=True,
                       help="Input JSON file path, e.g.: /path/to/robovqa_reasoning_0.json")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory, default: data/preprocess_data/cosmos_sft_data")
    parser.add_argument("--convert_ratio", type=float, default=1.0,
                       help="Conversion ratio, for debugging, e.g. 0.1 means convert 10% of data")
    parser.add_argument("--video_root", type=str, default=None,
                       help="Video file root directory, for building absolute paths. If not specified, use original relative paths")

    args = parser.parse_args()

    # Set default output directory
    if args.output_dir is None:
        # Infer dataset subset name from input file path
        input_path = Path(args.input_file)
        parent_dir = input_path.parent.name
        dataset_name = "cosmos-r1-sft"
        output_dir = f"data/preprocessed_data/{dataset_name}/{parent_dir}"
    else:
        output_dir = args.output_dir

    # Execute conversion
    convert_cosmos_to_msswift(
        input_file=args.input_file,
        output_dir=output_dir,
        convert_ratio=args.convert_ratio,
        video_root=args.video_root
    )

if __name__ == "__main__":
    main()
