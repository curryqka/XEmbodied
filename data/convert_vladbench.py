# UPDATE: Replace placeholder paths with your actual paths.
import json
import os
import argparse
import random
from glob import glob
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from pathlib import Path

def get_image_paths(item: Dict[str, Any], json_path: str) -> tuple:
    """
    Get image paths from item, handling both single and multiple images.
    
    Args:
        item: The data item
        json_path: Path to the JSON file
        
    Returns:
        Tuple of (gt_images, gt_image_plot_images)
    """
    # Get base URL for image paths
    base_url = json_path.replace('_E.json', '')
    
    if 'sequence' in item.keys():
        base_url = os.path.join(base_url, item['sequence'])
    
    # Handle regular image paths
    if isinstance(item.get('image_path'), str):
        gt_images = [os.path.join(base_url, item['image_path'])]
    elif isinstance(item.get('image_path'), list):
        gt_images = [os.path.join(base_url, img_path) for img_path in item['image_path']]
    else:
        gt_images = []
    
    # Handle plot image paths
    if 'image_path_plot' in item.keys():
        if isinstance(item['image_path_plot'], str):
            gt_image_plot_images = [os.path.join(base_url, item['image_path_plot'])]
        elif isinstance(item['image_path_plot'], list):
            gt_image_plot_images = [os.path.join(base_url, img_path) for img_path in item['image_path_plot']]
        else:
            gt_image_plot_images = []
    else:
        gt_image_plot_images = []
    
    return gt_images, gt_image_plot_images

def check_image_exists(image_path: str, warning_list: List[str], base_url: str = None) -> bool:
    """
    Check if an image file exists and log warning if it doesn't.
    
    Args:
        image_path: Path to the image file
        warning_list: List to store warning messages
        base_url: Base URL for constructing relative paths in warnings
        
    Returns:
        True if image exists, False otherwise
    """
    if not os.path.exists(image_path):
        # Create relative path for warning message
        if base_url:
            rel_path = os.path.relpath(image_path, base_url)
        else:
            rel_path = image_path
            
        warning_list.append(f"Image not found: {rel_path}")
        return False
    return True

def process_question_with_image_paths(question: str, item: Dict[str, Any], base_url: str, warning_list: List[str]) -> Tuple[str, List[str]]:
    """
    Process a single question to extract image paths and the actual question text.
    Handles cases where questions contain image paths.
    
    Args:
        question: The question string, may contain image paths
        item: The data item
        base_url: Base URL for constructing full image paths
        warning_list: List to store warning messages
        
    Returns:
        Tuple of (processed_question, image_paths_list)
    """
    # Check if question contains explicit image paths (pattern: image.png; question)
    if ';' in question:
        parts = question.split(';', 1)
        if len(parts) == 2:
            image_part = parts[0].strip()
            question_text = parts[1].strip()
            
            # Check if the image part looks like a file path (contains .png, .jpg, etc.)
            image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif']
            if any(image_part.lower().endswith(ext) for ext in image_extensions):
                # This is a direct image path
                image_path = os.path.join(base_url, image_part)
                # Check if image exists
                if check_image_exists(image_path, warning_list, base_url):
                    image_paths = [image_path]
                else:
                    # Image not found, we'll skip this question
                    return None, []
                # Construct question with image placeholder
                processed_question = '<image>' + question_text
                return processed_question, image_paths
    
    # Check for placeholders
    if '[image_path_plot]; ' in question:
        # Use plot images
        if isinstance(item.get('image_path_plot'), str):
            image_path = os.path.join(base_url, item['image_path_plot'])
            if check_image_exists(image_path, warning_list, base_url):
                image_paths = [image_path]
            else:
                return None, []
        elif isinstance(item.get('image_path_plot'), list):
            image_paths = []
            for img_path in item['image_path_plot']:
                full_path = os.path.join(base_url, img_path)
                if check_image_exists(full_path, warning_list, base_url):
                    image_paths.append(full_path)
            if not image_paths:
                return None, []
        else:
            return None, []
        processed_question = question.replace('[image_path_plot]; ', '<image>')
        return processed_question, image_paths
        
    elif '[image_path_plot];' in question:
        if isinstance(item.get('image_path_plot'), str):
            image_path = os.path.join(base_url, item['image_path_plot'])
            if check_image_exists(image_path, warning_list, base_url):
                image_paths = [image_path]
            else:
                return None, []
        elif isinstance(item.get('image_path_plot'), list):
            image_paths = []
            for img_path in item['image_path_plot']:
                full_path = os.path.join(base_url, img_path)
                if check_image_exists(full_path, warning_list, base_url):
                    image_paths.append(full_path)
            if not image_paths:
                return None, []
        else:
            return None, []
        processed_question = question.replace('[image_path_plot];', '<image>')
        return processed_question, image_paths
        
    elif '[image_path]; ' in question:
        if isinstance(item.get('image_path'), str):
            image_path = os.path.join(base_url, item['image_path'])
            if check_image_exists(image_path, warning_list, base_url):
                image_paths = [image_path]
            else:
                return None, []
        elif isinstance(item.get('image_path'), list):
            image_paths = []
            for img_path in item['image_path']:
                full_path = os.path.join(base_url, img_path)
                if check_image_exists(full_path, warning_list, base_url):
                    image_paths.append(full_path)
            if not image_paths:
                return None, []
        else:
            return None, []
        processed_question = question.replace('[image_path]; ', '<image>')
        return processed_question, image_paths
        
    elif '[image_path];' in question:
        if isinstance(item.get('image_path'), str):
            image_path = os.path.join(base_url, item['image_path'])
            if check_image_exists(image_path, warning_list, base_url):
                image_paths = [image_path]
            else:
                return None, []
        elif isinstance(item.get('image_path'), list):
            image_paths = []
            for img_path in item['image_path']:
                full_path = os.path.join(base_url, img_path)
                if check_image_exists(full_path, warning_list, base_url):
                    image_paths.append(full_path)
            if not image_paths:
                return None, []
        else:
            return None, []
        processed_question = question.replace('[image_path];', '<image>')
        return processed_question, image_paths
    
    # Default: no image placeholder found, use item's image_path
    if isinstance(item.get('image_path'), str):
        image_path = os.path.join(base_url, item['image_path'])
        if check_image_exists(image_path, warning_list, base_url):
            image_paths = [image_path]
        else:
            return None, []
    elif isinstance(item.get('image_path'), list):
        image_paths = []
        for img_path in item['image_path']:
            full_path = os.path.join(base_url, img_path)
            if check_image_exists(full_path, warning_list, base_url):
                image_paths.append(full_path)
        if not image_paths:
            return None, []
    else:
        return None, []
    
    # Add image placeholder at the beginning
    if image_paths:
        processed_question = '<image>' + question
    else:
        processed_question = question
    
    return processed_question, image_paths

def process_vladbench_data(json_path: str, root_dir: str, system_prompt: str = None, convert_ratio: float = 1.0) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Process a single VLADBench JSON file and convert it to the target format.
    
    Args:
        json_path: Path to the input JSON file
        root_dir: Root directory of VLADBench dataset
        system_prompt: System prompt to add to each conversation
        convert_ratio: Ratio of data to convert (for debugging)
        
    Returns:
        Tuple of (processed_entries, stats_dict)
    """
    processed_entries = []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # Statistics
    stats = {
        'total_items': len(original_data),
        'processed_questions': 0,
        'skipped_no_image': 0,
        'skipped_image_not_found': 0,
        'image_not_found_warnings': []
    }
    
    # Apply convert_ratio if specified
    if convert_ratio < 1.0:
        sample_size = max(1, int(len(original_data) * convert_ratio))
        original_data = random.sample(original_data, sample_size)
        stats['sampled_items'] = sample_size
    
    for item in tqdm(original_data, desc=f"Processing {os.path.basename(json_path)}"):
        # Get base URL for image paths
        base_url = json_path.replace('_E.json', '')
        if 'sequence' in item.keys():
            base_url = os.path.join(base_url, item['sequence'])
        
        # Get image paths from item (for fallback)
        gt_images, gt_image_plot_images = get_image_paths(item, json_path)
        
        # Extract metadata
        metadata = {k: v for k, v in item.items() if k not in ['questions', 'reference']}
        
        # Process each question
        gt_num = len(item['questions'])
        
        for i in range(gt_num):
            gt_q = item['questions'][i]
            qt_a = item['reference'][i]
            
            if not isinstance(gt_q, str):
                gt_q = str(gt_q)
            if not isinstance(qt_a, str):
                qt_a = str(qt_a)
            
            # Process the question to extract image paths and format it
            processed_question, image_paths = process_question_with_image_paths(
                gt_q, item, base_url, stats['image_not_found_warnings']
            )
            
            # Check if we got valid processed question and image paths
            if processed_question is None or not image_paths:
                stats['skipped_image_not_found'] += 1
                continue
            
            # Create message list
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
                
            # Add user and assistant messages
            messages.append({
                "role": "user",
                "content": processed_question.strip()
            })
            messages.append({
                "role": "assistant",
                "content": qt_a.strip()
            })
            
            # Create entry with images and metadata
            entry = {
                "messages": messages,
                "images": image_paths
            }
            
            # Add metadata
            if metadata:
                # entry["meta_data"] = metadata
                pass
            
            processed_entries.append(entry)
            stats['processed_questions'] += 1
    
    return processed_entries, stats

def save_jsonl(data: List[Dict[str, Any]], output_path: str):
    """Save data to JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for entry in data:
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"Saved {len(data)} entries to {output_path}")

def find_vladbench_files(root_dir: str) -> List[str]:
    """
    Find all _E.json files in the VLADBench directory structure.
    
    Args:
        root_dir: Root directory of VLADBench dataset
        
    Returns:
        List of paths to all _E.json files
    """
    # Define the pattern to search for E.json files
    pattern = os.path.join(root_dir, "**", "*_E.json")
    json_files = glob(pattern, recursive=True)
    
    # Filter to only include files in expected categories
    expected_paths = [
        # Ego_Decision_Planning
        "Ego_Decision_Planning/Ego_Action_Reasoning/Drive_Efficiency_E.json",
        "Ego_Decision_Planning/Ego_Action_Reasoning/Key_Obsturction_Detection_E.json",
        "Ego_Decision_Planning/Ego_Action_Reasoning/Risk_Prediction_E.json",
        "Ego_Decision_Planning/Ego_Action_Reasoning/Spatial_Temporal_Reasoning_E.json",
        "Ego_Decision_Planning/Meta_Action_Decision/Lateral_E.json",
        "Ego_Decision_Planning/Meta_Action_Decision/Longitudinal_E.json",
        # General_Element_Recognition
        "General_Element_Recognition/Background/Light_E.json",
        "General_Element_Recognition/Background/Weather_E.json",
        "General_Element_Recognition/Foreground/Lane_Recognition_E.json",
        "General_Element_Recognition/Foreground/Obstruction_Recognition_E.json",
        "General_Element_Recognition/Foreground/Vehicle_Recognition_E.json",
        "General_Element_Recognition/Foreground/Vehicle_Status_E.json",
        "General_Element_Recognition/Foreground/VRU_Recognition_E.json",
        # Target_Attribute_Comprehension
        "Target_Attribute_Comprehension/Behavior_Understanding/Vehicle_Bahavior_E.json",
        "Target_Attribute_Comprehension/Behavior_Understanding/VRU_Bahavior_E.json",
        "Target_Attribute_Comprehension/Intention_Judgment/Long_Short_Parking_E.json",
        "Target_Attribute_Comprehension/Intention_Judgment/Vehicle_Cutin_E.json",
        "Target_Attribute_Comprehension/Intention_Judgment/VRU_Cross_E.json",
        "Target_Attribute_Comprehension/Intention_Judgment/VRU_Cutin_E.json",
        # Traffic_Graph_Generation
        "Traffic_Graph_Generation/Lane_Element_Relation/Lane_Change_Relation_E.json",
        "Traffic_Graph_Generation/Lane_Element_Relation/Lane_Speed_Relation_E.json",
        "Traffic_Graph_Generation/Signal_Element_Relation/Light_Lane_Relation_E.json",
        "Traffic_Graph_Generation/Signal_Element_Relation/Sign_Lane_Relation_E.json",
        "Traffic_Graph_Generation/Signal_Element_Relation/Sign_Sign_Relation_E.json",
        # Traffic_Knowledge_Understanding
        "Traffic_Knowledge_Understanding/Road_Passage_Provisions/Right_Of_Way_E.json",
        "Traffic_Knowledge_Understanding/Road_Traffic_Signals/Pavement_Marking_E.json",
        "Traffic_Knowledge_Understanding/Road_Traffic_Signals/Traffic_Light_E.json",
        "Traffic_Knowledge_Understanding/Road_Traffic_Signals/Traffic_Sign_E.json",
    ]
    
    # Get all expected full paths
    expected_full_paths = [os.path.join(root_dir, path) for path in expected_paths]
    
    # Filter found files to only include expected ones
    found_files = []
    for json_file in json_files:
        # Normalize path for comparison
        normalized = os.path.normpath(json_file)
        if normalized in expected_full_paths:
            found_files.append(json_file)
    
    print(f"Found {len(found_files)} VLADBench JSON files:")
    for file in found_files:
        print(f"  - {os.path.relpath(file, root_dir)}")
    
    return found_files

def main():
    parser = argparse.ArgumentParser(description='Convert VLADBench dataset to MS-Swift training format')
    parser.add_argument('--input_file', type=str, default=None,
                       help='Single input JSON file (if not specified, processes all files in root_dir)')
    parser.add_argument('--root_dir', type=str, default=None,
                       help='Root directory of VLADBench dataset (required if input_file not specified)')
    parser.add_argument('--output_dir', type=str, default="data/preprocess_data/VLADBench",
                       help='Output directory for JSONL files (default: data/preprocess_data/VLADBench)')
    parser.add_argument('--convert_ratio', type=float, default=1.0,
                       help='Ratio of data to convert (for debugging, default: 1.0)')
    parser.add_argument('--system_prompt', type=str, 
                       default='You are a specialized model for autonomous driving scene understanding. You are designed to analyze and understand various elements in driving scenarios, including but not limited to: road conditions, traffic participants, traffic rules, decision making, and risk assessment.',
                       help='System prompt to add to each conversation')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input_file is None and args.root_dir is None:
        parser.error("Either --input_file or --root_dir must be specified")
    
    # Create output directory
    dataset_name = "VLADBench"
    if args.output_dir == "data/preprocess_data/VLADBench":
        args.output_dir = f"data/preprocess_data/{dataset_name}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find input files
    if args.input_file:
        json_files = [args.input_file]
    else:
        json_files = find_vladbench_files(args.root_dir)
    
    if not json_files:
        print(f"No VLADBench JSON files found")
        return
    
    all_processed_data = []
    all_stats = []
    all_warnings = []
    
    # Process each JSON file
    for json_path in json_files:
        # Get relative path for naming
        if args.root_dir:
            rel_path = os.path.relpath(json_path, args.root_dir)
        else:
            rel_path = os.path.basename(json_path)
        file_name = os.path.splitext(os.path.basename(json_path))[0]
        
        print(f"\n{'='*60}")
        print(f"Processing {rel_path}...")
        print(f"Convert ratio: {args.convert_ratio}")
        
        # Process the JSON file
        processed_data, stats = process_vladbench_data(
            json_path, 
            args.root_dir if args.root_dir else os.path.dirname(json_path), 
            args.system_prompt, 
            args.convert_ratio
        )
        
        # Add filename to stats
        stats['filename'] = os.path.basename(json_path)
        stats['rel_path'] = rel_path
        all_stats.append(stats)
        all_warnings.extend(stats['image_not_found_warnings'])
        
        if not processed_data:
            print(f"No data processed from {rel_path}")
            continue
        
        # Count multi-image entries
        multi_image_count = sum(1 for entry in processed_data if len(entry.get('images', [])) > 1)
        print(f"\nFile Statistics for {rel_path}:")
        print(f"  - Total items in file: {stats['total_items']}")
        if 'sampled_items' in stats:
            print(f"  - Items after sampling: {stats['sampled_items']}")
        print(f"  - Processed questions: {stats['processed_questions']}")
        print(f"  - Skipped (image not found): {stats['skipped_image_not_found']}")
        print(f"  - Single-image entries: {len(processed_data) - multi_image_count}")
        print(f"  - Multi-image entries: {multi_image_count}")
        if stats['image_not_found_warnings']:
            print(f"  - Image not found warnings: {len(stats['image_not_found_warnings'])}")
        
        # Save individual JSONL file
        output_name = f"{file_name}.jsonl"
        output_path = os.path.join(args.output_dir, output_name)
        save_jsonl(processed_data, output_path)
        
        # Add to combined data
        all_processed_data.extend(processed_data)
    
    # Save combined data
    if all_processed_data:
        combined_output_path = os.path.join(args.output_dir, f"{dataset_name.lower()}_combined.jsonl")
        save_jsonl(all_processed_data, combined_output_path)
        
        # Print overall statistics
        print(f"\n{'='*60}")
        print("OVERALL STATISTICS:")
        print(f"{'='*60}")
        
        total_files = len(all_stats)
        total_original_items = sum(s.get('total_items', 0) for s in all_stats)
        total_processed_questions = sum(s.get('processed_questions', 0) for s in all_stats)
        total_skipped_image_not_found = sum(s.get('skipped_image_not_found', 0) for s in all_stats)
        total_warnings = len(all_warnings)
        total_entries = len(all_processed_data)
        total_multi_image = sum(1 for entry in all_processed_data if len(entry.get('images', [])) > 1)
        total_with_meta = sum(1 for entry in all_processed_data if 'meta_data' in entry)
        
        print(f"Total files processed: {total_files}")
        print(f"Total original items: {total_original_items}")
        print(f"Total processed questions: {total_processed_questions}")
        print(f"Total skipped (image not found): {total_skipped_image_not_found}")
        print(f"Total image not found warnings: {total_warnings}")
        print(f"\nOutput dataset statistics:")
        print(f"  - Total entries: {total_entries}")
        print(f"  - Single-image entries: {total_entries - total_multi_image}")
        print(f"  - Multi-image entries: {total_multi_image}")
        print(f"  - Entries with metadata: {total_with_meta}")
        print(f"\nOutput directory: {args.output_dir}")
        print(f"Saved combined dataset to: {combined_output_path}")
        
        # Print warnings if any
        if all_warnings:
            print(f"\n{'='*60}")
            print(f"IMAGE NOT FOUND WARNINGS ({len(all_warnings)} total):")
            print(f"{'='*60}")
            # Show first 10 warnings
            for i, warning in enumerate(all_warnings[:10]):
                print(f"{i+1}. {warning}")
            if len(all_warnings) > 10:
                print(f"... and {len(all_warnings) - 10} more warnings")
            
            # Save all warnings to a file
            warnings_file = os.path.join(args.output_dir, "image_not_found_warnings.txt")
            with open(warnings_file, 'w', encoding='utf-8') as f:
                for warning in all_warnings:
                    f.write(f"{warning}\n")
            print(f"\nAll warnings saved to: {warnings_file}")
        
        # Save detailed statistics
        stats_file = os.path.join(args.output_dir, "conversion_statistics.json")
        stats_summary = {
            "total_files": total_files,
            "total_original_items": total_original_items,
            "total_processed_questions": total_processed_questions,
            "total_skipped_image_not_found": total_skipped_image_not_found,
            "total_image_not_found_warnings": total_warnings,
            "output_entries": total_entries,
            "single_image_entries": total_entries - total_multi_image,
            "multi_image_entries": total_multi_image,
            "entries_with_metadata": total_with_meta,
            "file_details": all_stats,
            "output_directory": args.output_dir,
            "combined_output_file": combined_output_path
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed statistics saved to: {stats_file}")
        
    else:
        print("No data was processed. Check if the input files are in the correct format.")
        
    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()