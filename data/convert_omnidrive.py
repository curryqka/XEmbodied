# UPDATE: Replace placeholder paths with your actual paths.
import pickle
import json
import os
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np

class OmniDriveConverter:
    def __init__(self, img_root: str = None, task_type: str = "vqa", data_split: str = "train"):
        """
        Initialize converter

        Args:
            img_root: Image root directory path
            task_type: Task type, 'vqa' or 'conv'
            data_split: Data split, 'train' or 'val'
        """
        self.img_root = img_root
        self.task_type = task_type
        self.data_split = data_split

        # Camera view mapping
        self.camera_views = {
            'CAM_FRONT': 'front',
            'CAM_FRONT_RIGHT': 'front right',
            'CAM_FRONT_LEFT': 'front left',
            'CAM_BACK': 'rear',
            'CAM_BACK_RIGHT': 'rear right',
            'CAM_BACK_LEFT': 'rear left'
        }

        # Class mapping
        self.class_mapping = {
            0: 'car',
            1: 'truck',
            2: 'bus',
            3: 'trailer',
            4: 'construction vehicle',
            5: 'pedestrian',
            6: 'motorcycle',
            7: 'bicycle',
            8: 'traffic cone',
            9: 'barrier',
            10: 'other'
        }

        # System prompt
        if task_type == "vqa":
            self.system_prompt = "You are an autonomous driving assistant. Analyze the driving scene from the six camera views and answer the question about the scene."
        else:  # conv
            self.system_prompt = "You are an autonomous driving assistant. You need to understand the driving scene from multiple camera perspectives and have a conversation about the environment, objects, and driving situations."

    def get_image_paths(self, sample: Dict) -> List[str]:
        """
        Get image paths from sample

        Args:
            sample: Sample data

        Returns:
            Image path list
        """
        image_paths = []

        # Process six cameras in fixed order
        camera_order = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_RIGHT',
            'CAM_BACK_LEFT'
        ]

        for cam_name in camera_order:
            if cam_name in sample.get('images', {}):
                cam_data = sample['images'][cam_name]
                img_path = cam_data.get('img_path', '')

                # Build full image path
                if self.img_root and img_path:
                    # Images may be stored in samples directory
                    full_img_path = os.path.join(self.img_root, cam_name, img_path)
                else:
                    full_img_path = img_path

                image_paths.append(full_img_path)
            else:
                # If no camera data, add empty string
                image_paths.append("")

        return image_paths

    def get_user_prompt_with_images(self, sample: Dict, question: str = "") -> str:
        """
        Get user prompt with image tags

        Args:
            sample: Sample data
            question: Question text

        Returns:
            User prompt with image tags
        """
        # Build image tags
        image_tags = []
        camera_order = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_RIGHT',
            'CAM_BACK_LEFT'
        ]

        for cam_name in camera_order:
            view_name = self.camera_views.get(cam_name, cam_name)
            image_tags.append(f"<image> ({view_name} view)")

        image_prompts = "\n".join(image_tags)

        if self.task_type == "vqa":
            if question:
                return f"{image_prompts}\n\n{question}"
            else:
                return f"{image_prompts}\n\nDescribe the driving scene shown in the camera views."
        else:  # conv
            if question:
                return f"{image_prompts}\n\n{question}"
            else:
                return f"{image_prompts}\n\nLet's discuss the driving scene shown in the camera views."

    def convert_sample_vqa(self, sample: Dict, idx: int) -> List[Dict]:
        """
        Convert single sample to VQA format

        Args:
            sample: Original sample data
            idx: Sample index

        Returns:
            List of converted VQA format data
        """
        # Get image paths
        image_paths = self.get_image_paths(sample)

        # Generate multiple VQA questions from annotations
        qa_pairs = self.generate_qa_from_instances(sample)

        # If no questions generated, use default question
        if not qa_pairs:
            qa_pairs = [{
                "question": "Describe the driving scene shown in the camera views.",
                "answer": "The scene shows a driving environment with multiple camera perspectives around the vehicle."
            }]

        # Create a separate sample for each QA pair
        converted_samples = []

        for qa in qa_pairs:
            # Build user prompt
            user_prompt = self.get_user_prompt_with_images(sample, qa["question"])

            # Build conversation messages - single turn VQA
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": qa["answer"]}
            ]

            # Build final output
            output = {
                "messages": messages,
                "images": image_paths,
                "metadata": {
                    "sample_idx": sample.get('sample_idx', idx),
                    "token": sample.get('token', ''),
                    "timestamp": sample.get('timestamp', 0),
                    "question_type": qa.get("type", "general"),
                    "task_type": "vqa",
                    "data_split": self.data_split
                }
            }

            converted_samples.append(output)

        return converted_samples

    def convert_sample_conv(self, sample: Dict, idx: int, conv_data: List[Dict] = None) -> Dict:
        """
        Convert single sample to conversation format

        Args:
            sample: Original sample data
            idx: Sample index
            conv_data: Conversation data

        Returns:
            Converted conversation format data
        """
        # Get image paths
        image_paths = self.get_image_paths(sample)

        # Build conversation messages
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        # Add conversation turns
        if conv_data and isinstance(conv_data, list):
            for i, qa in enumerate(conv_data):
                if i == 0:
                    # First turn includes images
                    user_prompt = self.get_user_prompt_with_images(sample, qa.get("question", ""))
                else:
                    # Subsequent turns are text only
                    user_prompt = qa.get("question", "")

                messages.append({"role": "user", "content": user_prompt})
                messages.append({"role": "assistant", "content": qa.get("answer", "")})
        else:
            # If no conversation data, create default conversation
            user_prompt = self.get_user_prompt_with_images(sample, "Describe the driving scene.")
            messages.extend([
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "The scene shows a driving environment with multiple camera perspectives around the vehicle."}
            ])

        # Build final output
        output = {
            "messages": messages,
            "images": image_paths,
            "metadata": {
                "sample_idx": sample.get('sample_idx', idx),
                "token": sample.get('token', ''),
                "timestamp": sample.get('timestamp', 0),
                "num_turns": (len(messages) - 1) // 2,  # Subtract system message
                "task_type": "conversation",
                "data_split": self.data_split
            }
        }

        return output

    def convert_sample(self, sample: Dict, idx: int, conv_dir: str = None) -> Union[List[Dict], Dict]:
        """
        Convert single sample

        Args:
            sample: Original sample data
            idx: Sample index
            conv_dir: Conversation data directory (only for conv task)

        Returns:
            Converted data
        """
        if self.task_type == "vqa":
            return self.convert_sample_vqa(sample, idx)
        else:  # conv
            # Try to load conversation data
            conv_data = None
            token = sample.get('token', '')

            if conv_dir and token:
                conv_path = os.path.join(conv_dir, f"{token}.json")
                if os.path.exists(conv_path):
                    try:
                        with open(conv_path, 'r', encoding='utf-8') as f:
                            conv_data = json.load(f)
                    except Exception as e:
                        print(f"Error loading conversation data from {conv_path}: {e}")

            return self.convert_sample_conv(sample, idx, conv_data)

    def generate_qa_from_instances(self, sample: Dict) -> List[Dict]:
        """
        Generate VQA questions from instance annotations

        Args:
            sample: Sample data

        Returns:
            VQA question list
        """
        qa_pairs = []
        instances = sample.get('instances', [])

        if not instances:
            return qa_pairs

        # 1. Object count questions
        class_counts = {}
        for inst in instances:
            label = inst.get('bbox_label_3d', inst.get('bbox_label', -1))
            class_name = self.class_mapping.get(label, f"class_{label}")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        if class_counts:
            class_desc = ", ".join([f"{count} {cls}" for cls, count in class_counts.items()])
            total_objects = sum(class_counts.values())

            qa_pairs.append({
                "question": f"How many objects are detected in the scene?",
                "answer": f"There are {total_objects} detected objects in the scene.",
                "type": "object_count"
            })

            qa_pairs.append({
                "question": f"What types of objects are detected and how many of each?",
                "answer": f"The detected objects are: {class_desc}.",
                "type": "object_types"
            })

        # 2. Vehicle related questions
        vehicles = [inst for inst in instances if inst.get('bbox_label_3d') in [0, 1, 2, 3, 4]]
        if vehicles:
            qa_pairs.append({
                "question": f"How many vehicles are detected in the scene?",
                "answer": f"There are {len(vehicles)} vehicles detected.",
                "type": "vehicle_count"
            })

            # If there are vehicles, ask about distance info
            if vehicles and 'bbox_3d' in vehicles[0]:
                distances = []
                for vehicle in vehicles:
                    bbox_3d = vehicle.get('bbox_3d', [])
                    if len(bbox_3d) >= 3:
                        # Calculate distance (x, y, z)
                        distance = np.sqrt(bbox_3d[0]**2 + bbox_3d[1]**2)
                        distances.append(f"{distance:.1f}m")

                if distances:
                    qa_pairs.append({
                        "question": f"What are the approximate distances of the detected vehicles?",
                        "answer": f"The vehicles are at approximate distances: {', '.join(distances[:3])} from the ego vehicle.",
                        "type": "vehicle_distance"
                    })

        # 3. Pedestrian/cyclist questions
        vulnerable = [inst for inst in instances if inst.get('bbox_label_3d') in [5, 6, 7]]
        if vulnerable:
            types = []
            for inst in vulnerable:
                label = inst.get('bbox_label_3d', -1)
                if label == 5:
                    types.append("pedestrian")
                elif label == 6:
                    types.append("motorcyclist")
                elif label == 7:
                    types.append("cyclist")

            type_counts = {}
            for t in types:
                type_counts[t] = type_counts.get(t, 0) + 1

            if type_counts:
                desc = ", ".join([f"{count} {typ}(s)" for typ, count in type_counts.items()])
                qa_pairs.append({
                    "question": f"Are there any pedestrians or cyclists in the scene?",
                    "answer": f"Yes, there are {desc} in the scene.",
                    "type": "vulnerable_users"
                })

        # 4. Camera view questions
        cam_instances = sample.get('cam_instances', {})
        if cam_instances:
            # Find camera that sees the most objects
            max_cam = None
            max_count = 0
            for cam, cams_inst in cam_instances.items():
                if len(cams_inst) > max_count:
                    max_count = len(cams_inst)
                    max_cam = cam

            if max_cam:
                view_name = self.camera_views.get(max_cam.replace('CAM_', ''), max_cam)
                qa_pairs.append({
                    "question": f"Which camera view shows the most objects?",
                    "answer": f"The {view_name} camera shows the most objects with {max_count} detected instances.",
                    "type": "camera_perspective"
                })

        # 5. Scene description questions
        qa_pairs.append({
            "question": f"Describe the overall driving scene based on the six camera views.",
            "answer": self.generate_scene_description(instances, cam_instances),
            "type": "scene_description"
        })

        return qa_pairs

    def generate_scene_description(self, instances, cam_instances) -> str:
        """Generate scene description"""
        if not instances:
            return "The scene appears to be relatively clear with no detected objects in the immediate vicinity."

        # Count main object types
        vehicle_count = len([inst for inst in instances if inst.get('bbox_label_3d') in [0, 1, 2, 3, 4]])
        pedestrian_count = len([inst for inst in instances if inst.get('bbox_label_3d') == 5])
        cyclist_count = len([inst for inst in instances if inst.get('bbox_label_3d') in [6, 7]])

        parts = []

        if vehicle_count > 0:
            parts.append(f"{vehicle_count} vehicle(s)")

        if pedestrian_count > 0:
            parts.append(f"{pedestrian_count} pedestrian(s)")

        if cyclist_count > 0:
            parts.append(f"{cyclist_count} cyclist(s)")

        if parts:
            object_desc = ", ".join(parts)
            return f"The driving scene contains {object_desc}. The environment appears to be an urban or suburban area with multiple road users."
        else:
            return "The driving scene shows a road environment with various objects detected from the six camera perspectives."

    def convert_dataset(self, data_list: List[Dict], convert_ratio: float = 1.0,
                        max_qa_per_sample: int = 3, conv_dir: str = None) -> List[Dict]:
        """
        Convert entire dataset

        Args:
            data_list: Original data list
            convert_ratio: Conversion ratio
            max_qa_per_sample: Maximum VQA questions per sample (VQA task only)
            conv_dir: Conversation data directory (Conversation task only)

        Returns:
            Converted data list
        """
        # Calculate number of samples to convert
        total_samples = len(data_list)
        convert_count = int(total_samples * convert_ratio)

        if convert_ratio < 1.0:
            # Random sampling
            indices = random.sample(range(total_samples), convert_count)
            selected_data = [data_list[i] for i in indices]
        else:
            selected_data = data_list

        converted_data = []
        skipped_no_conv = 0

        for idx, sample in enumerate(selected_data):
            try:
                if self.task_type == "vqa":
                    # VQA task: generate multiple VQA samples
                    converted_samples = self.convert_sample_vqa(sample, idx)

                    # Limit number of VQA questions per sample
                    if len(converted_samples) > max_qa_per_sample:
                        converted_samples = random.sample(converted_samples, max_qa_per_sample)

                    converted_data.extend(converted_samples)
                else:
                    # Conversation task: check if conversation data exists
                    token = sample.get('token', '')
                    if conv_dir and token:
                        conv_path = os.path.join(conv_dir, f"{token}.json")
                        if os.path.exists(conv_path):
                            converted = self.convert_sample(sample, idx, conv_dir)
                            converted_data.append(converted)
                        else:
                            skipped_no_conv += 1
                            if skipped_no_conv <= 5:  # Only print first 5 warnings
                                print(f"Warning: No conversation data found for token {token}")
                            continue
                    else:
                        # If no conversation directory, create a default conversation
                        converted = self.convert_sample_conv(sample, idx, None)
                        converted_data.append(converted)

                if (idx + 1) % 100 == 0:
                    if self.task_type == "vqa":
                        print(f"Converted {idx + 1}/{len(selected_data)} samples, generated {len(converted_data)} VQA samples")
                    else:
                        print(f"Converted {idx + 1}/{len(selected_data)} conversation samples")

            except Exception as e:
                print(f"Error converting sample {idx} (token: {sample.get('token', 'unknown')}): {e}")
                continue

        if self.task_type == "conv" and skipped_no_conv > 0:
            print(f"Skipped {skipped_no_conv} samples due to missing conversation data")

        return converted_data

def main():
    parser = argparse.ArgumentParser(description='Convert OmniDrive data to MS-Swift format')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input pickle file')
    parser.add_argument('--output_dir', type=str,
                       default='data/preprocess_data/OmniDrive',
                       help='Output directory for converted data')
    parser.add_argument('--task_type', type=str, choices=['vqa', 'conv'], default='vqa',
                       help='Task type: vqa (VQA format) or conv (conversation format)')
    parser.add_argument('--data_split', type=str, choices=['train', 'val'], default='train',
                       help='Data split: train or val')
    parser.add_argument('--convert_ratio', type=float, default=1.0,
                       help='Ratio of data to convert (0.0 to 1.0)')
    parser.add_argument('--img_root', type=str, default='',
                       help='Root directory for images (if paths in pickle are relative)')
    parser.add_argument('--conv_dir', type=str, default='',
                       help='Directory containing conversation JSON files (required for conv task)')
    parser.add_argument('--max_qa_per_sample', type=int, default=3,
                       help='Maximum number of VQA questions to generate per sample (for vqa task only)')
    parser.add_argument('--output_name', type=str, default='',
                       help='Output file name (default: auto-generated based on task type and data split)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set output filename
    if not args.output_name:
        if args.task_type == "vqa":
            args.output_name = f'omnidrive_vqa_{args.data_split}_ms_swift.jsonl'
        else:
            args.output_name = f'omnidrive_conversation_{args.data_split}_ms_swift.jsonl'

    # Load data
    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, 'rb') as f:
        data = pickle.load(f)

    # Extract data list
    if isinstance(data, dict) and 'data_list' in data:
        data_list = data['data_list']
        print(f"Loaded {len(data_list)} samples from data_list")
    elif isinstance(data, list):
        data_list = data
        print(f"Loaded {len(data_list)} samples")
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")

    # If conv_dir not provided, use default path
    if args.task_type == "conv" and not args.conv_dir:
        # Build default conv_dir path based on data_split
        base_conv_dir = "data/OmniDrive/data_nusc/conv"
        args.conv_dir = os.path.join(base_conv_dir, args.data_split)
        print(f"Using default conversation directory: {args.conv_dir}")

    # Check conversation directory
    if args.task_type == "conv" and args.conv_dir and not os.path.exists(args.conv_dir):
        print(f"Warning: Conversation directory {args.conv_dir} does not exist")
        # Try to create directory
        os.makedirs(args.conv_dir, exist_ok=True)
        print(f"Created conversation directory: {args.conv_dir}")

    # Initialize converter
    converter = OmniDriveConverter(img_root=args.img_root, task_type=args.task_type, data_split=args.data_split)

    # Convert data
    print(f"Converting {args.convert_ratio*100:.1f}% of data for {args.task_type} task ({args.data_split} split)...")
    converted_data = converter.convert_dataset(
        data_list,
        args.convert_ratio,
        args.max_qa_per_sample,
        args.conv_dir
    )

    # Save in JSON Lines format
    output_path = os.path.join(args.output_dir, args.output_name)
    print(f"Saving {len(converted_data)} samples to {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in converted_data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

    # Save as regular JSON format (optional)
    json_path = os.path.join(args.output_dir, args.output_name.replace('.jsonl', '.json'))
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)

    print(f"Conversion completed. Generated {len(converted_data)} samples.")
    print(f"Output saved to:")
    print(f"  - JSONL: {output_path}")
    print(f"  - JSON: {json_path}")

    # Output statistics
    if converted_data:
        print(f"\nTask type: {args.task_type}")
        print(f"Data split: {args.data_split}")

        if args.task_type == "vqa":
            # Count VQA question types
            question_types = {}
            for sample in converted_data:
                q_type = sample['metadata'].get('question_type', 'unknown')
                question_types[q_type] = question_types.get(q_type, 0) + 1

            print(f"VQA question type distribution:")
            for q_type, count in question_types.items():
                print(f"  {q_type}: {count}")

        else:  # conv
            # Count conversation turns
            turn_counts = {}
            for sample in converted_data:
                num_turns = sample['metadata'].get('num_turns', 0)
                turn_counts[num_turns] = turn_counts.get(num_turns, 0) + 1

            print(f"Conversation turn distribution:")
            for num_turns, count in turn_counts.items():
                print(f"  {num_turns} turns: {count} samples")

        # Show example
        print(f"\nExample sample:")
        sample = converted_data[0]
        if args.task_type == "vqa":
            print(f"  Question: {sample['messages'][1]['content'][-100:]}...")
            print(f"  Answer: {sample['messages'][2]['content'][:100]}...")
        else:
            print(f"  Number of messages: {len(sample['messages'])}")
            print(f"  First user message: {sample['messages'][1]['content'][:100]}...")
            print(f"  First assistant response: {sample['messages'][2]['content'][:100]}...")

if __name__ == '__main__':
    main()
