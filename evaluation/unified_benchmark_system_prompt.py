# UPDATE: Replace placeholder paths with your actual paths.
import os
import json

# Base directory path
BASE_DIR = "data/preprocess_data/all_eval_data"

# New system prompt content (English)
NEW_SYSTEM_PROMPT = (
    "You are a professional multimodal AI assistant. Watch the video, image or multi-view images and answer questions. "
    "Enclose your final answer within <answer> tags, like this: <answer>your answer here</answer>."
)

def update_jsonl_file(file_path):
    updated_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Find the message with role "system" in messages and replace content
            found_system = False
            for msg in data["messages"]:
                if msg["role"] == "system":
                    msg["content"] = NEW_SYSTEM_PROMPT
                    found_system = True
                    break
            if not found_system:
                # If no system message, insert one (though your data should have one)
                data["messages"].insert(0, {"role": "system", "content": NEW_SYSTEM_PROMPT})
            updated_lines.append(json.dumps(data, ensure_ascii=False))

    # Overwrite in place
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in updated_lines:
            f.write(line + '\n')

def main():
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith('.jsonl'):
                full_path = os.path.join(root, file)
                print(f"Processing: {full_path}")
                update_jsonl_file(full_path)
    print("All JSONL files updated successfully.")

if __name__ == "__main__":
    main()
