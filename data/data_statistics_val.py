# UPDATE: Replace placeholder paths with your actual paths.
# vabench, embodied-r1, roboafford, where2place, Part-Affordance-2K need to sample 20% into training set
# LingoQA-Action	data/zy_utils/lingoqa_action_train_qwen_v0.json
# Surds	data/zy_utils/merged_surds_vqas_v0.jsonl
# Maplmv2	data/MapLM/filter_data/maplm_filter_06.jsonl
# Maplmv2-Multi	data/MapLM/filter_data/maplm_filter_06_multi.jsonl
# Drivingvqa	data/Drivevqa/filter_data/drivevqa_rel_filter_06.jsonl
# DriveLMM-o1	data/DriveLMMo1/filter_data/filtered_raw_data_drivelmm_06_v2.jsonl
# OmniDrive-vqa	data/Omnidrive/filter_data/omnidrive_vqa_filtered_06.jsonl
# OmniDrive-conv	data/Omnidrive/filter_data/omnidrive_vqa_filtered_multi_06.jsonl
# Refcoco	data/RefCOCO/raw_data/refcoco_train_qwen_v1.jsonl
# Flickr	data/Flickr/filter_data/flickr_filter_06.jsonl
# Vqav2	data/VQAv2/filter_data/vqa_filter_06.jsonl
# Gqa	data/Gqa/filter_data/gqa_filter_06.jsonl
# Bdd	data/zy_utils/bdd100k_labels_images_train_weather_qwen_v1.json
# roborefit	data/Roborefit/filter_data/roborefit_filter_06.jsonl
# Cosmosr1	data/Cosmosr1/filter_data/cosmosr1_filter_06.jsonl
# vladbench	data/zy_utils/vladbench_train_image_rel_qwen_v1.json
# robovqa	data/zy_utils/robovqa_understanding.jsonl


import json
import random

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def save_subset_to_file(data, output_file_path):
    is_jsonl = output_file_path.lower().endswith('.jsonl')
    with open(output_file_path, 'w', encoding='utf-8') as file:
        if is_jsonl:
            for item in data:
                file.write(json.dumps(item) + '\n')
        else:
            json.dump(data, file, ensure_ascii=False, indent=4)

def sample_data(input_file, output_file, sample_rate=0.2):
    if input_file.lower().endswith('.json'):
        dataset = load_json_file(input_file)
    elif input_file.lower().endswith('.jsonl'):
        dataset = load_jsonl_file(input_file)
    else:
        raise ValueError("Unsupported file format")

    sampled_data = random.sample(dataset, int(len(dataset) * sample_rate))

    save_subset_to_file(sampled_data, output_file)

# Example usage:
datasets = [
    # Modify paths according to your actual situation
    # ("/path/to/your/dataset.json", "/path/to/output/train_set.json"),
    ("data/preprocess_data/all_eval_data/embodied-r1/embodied-r1_ms_swift_test.jsonl",
     "data/preprocess_data/embodied-r1-train/embodied-r1_ms_swift_train.jsonl"),
    ("data/preprocess_data/all_eval_data/Part-Affordance-2K/part_affordance_ms_swift.jsonl",
     "data/preprocess_data/Part-Affordance-2K-train/part_affordance_ms_swift_train.jsonl"),
    ("data/preprocess_data/all_eval_data/RoboAfford/RoboAfford_ms_swift.jsonl",
     "data/preprocess_data/RoboAfford-train/RoboAfford_ms_swift_train.jsonl"),
    ("data/preprocess_data/all_eval_data/vabench-point-bbox/vabench_bbox_absolute_no_reasoning_ms_swift.jsonl",
     "data/preprocess_data/vabench-point-bbox-train/vabench_bbox_absolute_no_reasoning_ms_swift_train.jsonl"),
    ("data/preprocess_data/all_eval_data/where2place/where2place_ms_swift_absolute.jsonl",
     "data/preprocess_data/where2place-train/where2place_ms_swift_absolute_train.jsonl")
]

for dataset, output in datasets:
    sample_data(dataset, output, 0.2)
