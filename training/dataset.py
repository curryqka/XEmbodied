# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional

from swift.llm import DatasetMeta, ResponsePreprocessor, MessagesPreprocessor, load_dataset, register_dataset


class CustomPreprocessor(ResponsePreprocessor):
    prompt = """Task: Based on the given two sentences, provide a similarity score between 0.0 and 5.0.
Sentence 1: {text1}
Sentence 2: {text2}
Similarity score: """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return super().preprocess({
            'query': self.prompt.format(text1=row['text1'], text2=row['text2']),
            'response': f"{row['label']:.1f}"
        })


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/stsb',
        hf_dataset_id='SetFit/stsb',
        preprocess_func=CustomPreprocessor(),
    ))



class BDDPreprocessor(MessagesPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:

        query = row.get('messages', '')[0]
        # breakpoint()
        assert query['role'] == 'user', 'this is not the user content!'
        # question_type = determine_question_type(query["content"])
        
        answer = row.get('messages', '')[1]
        assert answer['role'] =='assistant', 'this is not the assistant content!'
        
        channel = 'public_bdd100k'
        row.update({'solutions': answer, 'question_type': channel})
        return row
try:
    register_dataset(
        DatasetMeta(
            # ms_dataset_id='okwinds/clevr_cogen_a_train',
            # hf_dataset_id='ayeshaishaq/DriveLMMo1',
            dataset_path="/high_perf_store2/tfl-vepfs/zy/zy_utils/bdd100k_labels_images_train_weather_qwen_v1.json",
            preprocess_func=BDDPreprocessor(),
            tags=['qa', 'drive', 'vision', 'grpo']))
except:
    print(f'The `bdd-100k` has already been registered in the DATASET_MAPPING.')



# ----------------------------
# 2. VLAD-Bench Preprocessor
# ----------------------------
class VLADBenchPreprocessor(MessagesPreprocessor):
    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        messages = row.get('messages', [])
        assert len(messages) >= 2, "Messages must contain at least user and assistant."
        query = messages[0]
        answer = messages[1]
        assert query['role'] == 'user', 'First message must be user.'
        assert answer['role'] == 'assistant', 'Second message must be assistant.'
        
        channel = 'public_vladbench'
        row.update({'solutions': answer, 'question_type': channel})
        return row

# Register VLAD-Bench
try:
    register_dataset(
        DatasetMeta(
            dataset_path="/high_perf_store2/tfl-vepfs/zy/zy_utils/vladbench_train_image_qwen_v1.json",
            preprocess_func=VLADBenchPreprocessor(),
            tags=['qa', 'drive', 'vision', 'grpo']))
except Exception as e:
    print(f'VLAD-Bench already registered or error: {e}')

# ----------------------------
# 3. SURDS Preprocessor
# ----------------------------
class SURDSPreprocessor(MessagesPreprocessor):
    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        messages = row.get('messages', [])
        assert len(messages) >= 2, "Messages must contain at least user and assistant."
        query = messages[0]
        answer = messages[1]
        assert query['role'] == 'user', 'First message must be user.'
        assert answer['role'] == 'assistant', 'Second message must be assistant.'
        
        channel = 'public_surds'
        row.update({'solutions': answer, 'question_type': channel})
        return row

# Register SURDS
try:
    register_dataset(
        DatasetMeta(
            dataset_path="/high_perf_store/mlinfra-vepfs/qiankangan/Drive-MLLM-2/eval_vqas_reasoning_withjpgimage_train/swift_vqas/merged_surds_vqas_v0.jsonl",
            preprocess_func=SURDSPreprocessor(),
            tags=['qa', 'drive', 'vision', 'grpo']))
except Exception as e:
    print(f'SURDS already registered or error: {e}')


# ----------------------------
# 4. LingoQA Scenery Preprocessor
# ----------------------------
class LingoQASceneryPreprocessor(MessagesPreprocessor):
    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        messages = row.get('messages', [])
        assert len(messages) >= 2, "Messages must contain at least user and assistant."
        query = messages[0]
        answer = messages[1]
        assert query['role'] == 'user', 'First message must be user.'
        assert answer['role'] == 'assistant', 'Second message must be assistant.'
        
        channel = 'public_lingoqa_scenery'
        row.update({'solutions': answer, 'question_type': channel})
        return row


# ----------------------------
# 5. LingoQA Action Preprocessor
# ----------------------------
class LingoQAActionPreprocessor(MessagesPreprocessor):
    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        messages = row.get('messages', [])
        assert len(messages) >= 2, "Messages must contain at least user and assistant."
        query = messages[0]
        answer = messages[1]
        assert query['role'] == 'user', 'First message must be user.'
        assert answer['role'] == 'assistant', 'Second message must be assistant.'
        
        channel = 'public_lingoqa_action'
        row.update({'solutions': answer, 'question_type': channel})
        return row


# Register LingoQA Scenery
try:
    register_dataset(
        DatasetMeta(
            dataset_path="/high_perf_store2/tfl-vepfs/zy/zy_utils/lingoqa_scenery_train_qwen_v0.json",
            preprocess_func=LingoQASceneryPreprocessor(),
            tags=['qa', 'drive', 'vision', 'grpo']))
except Exception as e:
    print(f'LingoQA Scenery already registered or error: {e}')

# Register LingoQA Action
try:
    register_dataset(
        DatasetMeta(
            dataset_path="/high_perf_store2/tfl-vepfs/zy/zy_utils/lingoqa_action_train_qwen_v0.json",
            preprocess_func=LingoQAActionPreprocessor(),
            tags=['qa', 'drive', 'vision', 'grpo']))
except Exception as e:
    print(f'LingoQA Action already registered or error: {e}')

if __name__ == '__main__':
    dataset = load_dataset(['swift/stsb'])[0]
    print(f'dataset: {dataset}')
    print(f'dataset[0]: {dataset[0]}')
