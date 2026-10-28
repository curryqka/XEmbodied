# UPDATE: Replace placeholder paths with your actual paths.
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import glob

def get_concat_image_path(json_file_name: str, data_item: Dict) -> str:
    """
    Build concatenated image path based on JSON filename and data item

    Args:
        json_file_name: JSON filename (e.g. "biterror.json")
        data_item: Data item, containing scene_token and frame_token

    Returns:
        Full path to the concatenated image
    """
    # Extract type from JSON filename (e.g. "biterror")
    corruption_type = Path(json_file_name).stem
    scene_token = data_item.get("scene_token", "")
    frame_token = data_item.get("frame_token", "")

    if not scene_token or not frame_token:
        return ""

    # Build concatenated image path
    base_dir = "data/qiankangan/DriveLMM-o1-main/data/image2concat_drivebench"
    concat_image_name = f"{scene_token}_{frame_token}.png"
    concat_image_path = os.path.join(base_dir, corruption_type, concat_image_name)

    return concat_image_path

def convert_drivebench_json_to_ms_swift(input_file: str, output_file: str) -> None:
    """
    Convert original DriveBench JSON format to MS-Swift format JSONL

    Original data format:
    [
        {
            "scene_token": "da41ecbc644b4915b84bb732e35ebf8c",
            "frame_token": "7e4c3282bc2a4402b5d1d6705f9eb844",
            "question_type": "robust_qas",
            "question": "What is the current type of corruption?: A. Motion blur. B. Bit error. C. H256 ABR compression. D. Fog",
            "answer": "B. Bit error",
            "tag": [0],
            "image_path": {...}  # 6 camera paths
        },
        ...
    ]

    MS-Swift format:
    {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"}
        ],
        "images": ["concatenated image path"],
        "question_type": "robust_qas",
        "answer_index": [0],
        "scene_token": "...",
        "frame_token": "...",
        ...
    }
    """
    json_file_name = os.path.basename(input_file)

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        try:
            # Read entire JSON array
            data_list = json.load(infile)

            if not isinstance(data_list, list):
                print(f"Error: {input_file} is not a JSON array format")
                return

            print(f"Processing {json_file_name}: found {len(data_list)} samples")

            processed_count = 0
            for i, data_item in enumerate(data_list, 1):
                try:
                    # Get concatenated image path
                    concat_image_path = get_concat_image_path(json_file_name, data_item)

                    if not concat_image_path:
                        print(f"Warning: item {i} missing scene_token or frame_token")
                        continue

                    # Build MS-Swift format
                    ms_swift_data = {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an excellent driver. Analyze the 6-view concatenated images and answer the driving-related questions accordingly."
                            },
                            {
                                "role": "user",
                                "content": data_item.get("question", "")
                            },
                            {
                                "role": "assistant",
                                "content": data_item.get("answer", "")
                            }
                        ],
                        "images": [concat_image_path]  # Use concatenated image path
                    }

                    # Add reserved fields
                    reserved_keys = ["question_type", "tag", "scene_token", "frame_token"]
                    for key in reserved_keys:
                        if key in data_item:
                            if key == "tag":
                                ms_swift_data["answer_index"] = data_item[key]
                            else:
                                ms_swift_data[key] = data_item[key]

                    # Add other non-reserved fields (excluding already processed fields)
                    excluded_keys = ["question", "answer", "image_path"]
                    for key, value in data_item.items():
                        if key not in excluded_keys and key not in ms_swift_data:
                            ms_swift_data[key] = value

                    # Write output file
                    outfile.write(json.dumps(ms_swift_data, ensure_ascii=False) + "\n")
                    processed_count += 1

                except Exception as e:
                    print(f"Error: failed processing item {i} of {json_file_name}: {e}")
                    continue

            print(f"Done: successfully processed {processed_count}/{len(data_list)} samples")

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {input_file}: {e}")
        except Exception as e:
            print(f"Processing error: {input_file}: {e}")

def process_directory(input_dir: str, output_dir: str) -> None:
    """
    Process all non-final JSON files in directory

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all JSON files without final suffix
    json_files = []
    for json_file in glob.glob(os.path.join(input_dir, "*.json")):
        if "final" not in json_file.lower():
            json_files.append(json_file)

    if not json_files:
        print(f"No qualifying JSON files found in directory {input_dir}")
        return

    print(f"Found {len(json_files)} non-final JSON files:")
    for file_path in json_files:
        print(f"  - {os.path.basename(file_path)}")

    # Process each file
    for input_file in json_files:
        filename = os.path.basename(input_file)
        output_filename = f"{Path(filename).stem}_ms_swift_val.jsonl"
        output_file = os.path.join(output_dir, output_filename)

        print(f"\nProcessing file: {filename}")
        print(f"Output to: {output_filename}")

        convert_drivebench_json_to_ms_swift(input_file, output_file)

        # Verify conversion result
        with open(output_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line:
                try:
                    sample_data = json.loads(first_line)
                    print(f"Conversion successful! Example output:")
                    print(f"  messages: {len(sample_data.get('messages', []))} entries")
                    print(f"  images: {len(sample_data.get('images', []))} images")
                    print(f"  Image path: {sample_data.get('images', [])[0] if sample_data.get('images') else 'none'}")
                    print(f"  Other fields: {list(sample_data.keys() - {'messages', 'images'})}")
                except Exception as e:
                    print(f"Verification failed: {e}")
            else:
                print("Warning: output file is empty")

def process_single_file(input_path: str, output_dir: str = None) -> None:
    """
    Process a specified single file

    Args:
        input_path: Input file path
        output_dir: Output directory path (if None, output to same directory)
    """
    if not os.path.exists(input_path):
        print(f"File does not exist: {input_path}")
        return

    filename = os.path.basename(input_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{Path(filename).stem}_ms_swift.jsonl")
    else:
        input_dir = os.path.dirname(input_path)
        output_file = os.path.join(input_dir, f"{Path(filename).stem}_ms_swift.jsonl")

    print(f"\nProcessing file: {input_path}")
    print(f"Output to: {output_file}")

    convert_drivebench_json_to_ms_swift(input_path, output_file)

    # Show before/after comparison
    print(f"\nBefore conversion example:")
    with open(input_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
        if data_list and isinstance(data_list, list) and len(data_list) > 0:
            sample = data_list[0]
            print(f"  scene_token: {sample.get('scene_token', 'N/A')}")
            print(f"  frame_token: {sample.get('frame_token', 'N/A')}")
            print(f"  question: {sample.get('question', 'N/A')[:50]}...")
            print(f"  answer: {sample.get('answer', 'N/A')[:50]}...")
            print(f"  image_path: {list(sample.get('image_path', {}).keys()) if sample.get('image_path') else 'none'}")

    print(f"\nAfter conversion example:")
    with open(output_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if first_line:
            sample_data = json.loads(first_line)
            print(f"  messages: user: {sample_data.get('messages', [{}])[0].get('content', 'N/A')[:50]}...")
            print(f"  messages: assistant: {sample_data.get('messages', [{}])[1].get('content', 'N/A')[:50]}...")
            print(f"  images: {sample_data.get('images', ['N/A'])[0]}")
            print(f"  Other fields: {list(sample_data.keys() - {'messages', 'images'})}")

if __name__ == "__main__":
    # Configure paths
    input_dir = "data/preprocess_data/data/DriveBench20K"
    output_dir = "data/preprocess_data/all_eval_data/drivebench"

    print("=" * 60)
    print("DriveBench JSON to MS-Swift format converter")
    print("=" * 60)

    # Method 1: Process entire directory
    print("Batch processing entire directory (non-final files)")
    print("=" * 60)

    if os.path.exists(input_dir):
        process_directory(input_dir, output_dir)
    else:
        print(f"Input directory does not exist: {input_dir}")

    # Method 2: Process a specified single file
    print("\n" + "=" * 60)
    print("Process a specified single file (example)")
    print("=" * 60)

    # Uncomment and specify the single file to process here
    # specific_file = "data/preprocess_data/data/biterror.json"
    # if os.path.exists(specific_file):
    #     process_single_file(specific_file, output_dir)
    # else:
    #     print(f"Example file does not exist: {specific_file}")

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)

    # Output list of processed files
    print("\nProcessed file types:")
    print("biterror.json")
    print("bright.json")
    print("camcrash.json")
    print("clean.json")
    print("colorquant.json")
    print("fog.json")
    print("framelost.json")
    print("h256.json")
    print("lens.json")
    print("lowlight.json")
    print("motion.json")
    print("rain.json")
    print("saturate.json")
    print("snow.json")
    print("water.json")
    print("zoom.json")
    print("drivebench-test.json")
