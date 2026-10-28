# UPDATE: Replace placeholder paths with your actual paths.
# test_foundation_orm.py

import os
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict

# ----------------------------
# Step 1: Define Foundation_ORM (copy your class here for self-contained test)
# ----------------------------

import re
import torch.nn.functional as F
from sentence_transformers import util

class Foundation_ORM:
    def __init__(self, model_path="sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with HuggingFace model (auto-download if not local)."""

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()

    def __call__(
        self,
        completions: List[str],
        solutions: List[Dict[str, str]],
        question_type: List[str],
        **kwargs
    ) -> List[float]:
        assert len(completions) == len(solutions) == len(question_type), "Length mismatch"
        rewards = []
        for comp, sol, q_type in zip(completions, solutions, question_type):
            raw_gt = sol['content']
            raw_pred = comp
            gt_clean = self._extract_answer(raw_gt)
            pred_clean = self._extract_answer(raw_pred)

            if q_type in {'public_bdd100k', 'public_vladbench', 'public_surds'}:
                score = float(pred_clean.lower().strip() == gt_clean.lower().strip())
            elif q_type in {'public_lingoqa_scenery', 'public_lingoqa_action'}:
                score = self._compute_semantic_similarity(pred_clean, gt_clean)
            else:
                raise ValueError(f"Unsupported question_type: {q_type}")
            rewards.append(score)
        return rewards

    def _extract_answer(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else text.strip()

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.process_text_embed([text1])
        emb2 = self.process_text_embed([text2])
        sim = util.cos_sim(emb1, emb2).item()
        return max(0.0, min(1.0, sim))

    def process_text_embed(self, sentences: List[str]) -> torch.Tensor:
        encoded_input = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        device = next(self.model.parameters()).device
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        token_embeddings = model_output.last_hidden_state
        input_mask = encoded_input['attention_mask'].unsqueeze(-1).float()
        summed = torch.sum(token_embeddings * input_mask, dim=1)
        divisor = torch.clamp(input_mask.sum(dim=1), min=1e-9)
        embeddings = summed / divisor
        return F.normalize(embeddings, p=2, dim=1)


# ----------------------------
# Step 2: Define test cases
# ----------------------------

def run_tests():
    print("Starting offline tests for Foundation_ORM...\n")

    # Initialize ORM (will auto-download model if needed)
    orm = Foundation_ORM(model_path="checkpoints/all-MiniLM-L6-v2")

    # Test data
    test_cases = [
        # Case 1: BDD100K - exact match, single letter
        {
            "completions": ["E"],
            "solutions": [{"content": "E"}],
            "question_type": ["public_bdd100k"],
            "expected": [1.0],
            "desc": "BDD100K: Correct single-letter answer"
        },
        # Case 2: SURDS - with <answer> tag in GT, clean prediction
        {
            "completions": ["Between 21 meters and 28 meters"],
            "solutions": [{"content": "<answer>Between 21 meters and 28 meters</answer>"}],
            "question_type": ["public_surds"],
            "expected": [1.0],
            "desc": "SURDS: Clean match after tag extraction"
        },
        # Case 3: SURDS - both have tags (edge case)
        {
            "completions": ["<answer>Between 21 meters and 28 meters</answer>"],
            "solutions": [{"content": "<answer>Between 21 meters and 28 meters</answer>"}],
            "question_type": ["public_surds"],
            "expected": [1.0],
            "desc": "SURDS: Both sides wrapped in <answer>"
        },
        # Case 4: SURDS - wrong answer
        {
            "completions": ["Between 12 meters and 16 meters"],
            "solutions": [{"content": "<answer>Between 21 meters and 28 meters</answer>"}],
            "question_type": ["public_surds"],
            "expected": [0.0],
            "desc": "SURDS: Incorrect answer"
        },
        # Case 5: VLAD-Bench - lowercase tolerance
        {
            "completions": ["low"],
            "solutions": [{"content": "Low"}],
            "question_type": ["public_vladbench"],
            "expected": [1.0],
            "desc": "VLAD-Bench: Case-insensitive match"
        },
        # Case 6: LingoQA - semantic similarity (should be high)
        {
            "completions": ["I am turning left because the road curves that way."],
            "solutions": [{"content": "Turning left due to road curvature."}],
            "question_type": ["public_lingoqa_scenery"],
            "expected_range": [(0.7, 1.0)],  # Not exact, but high similarity
            "desc": "LingoQA-scenery: Semantic similarity (high)"
        },
        # Case 7: LingoQA - unrelated response (should be low)
        {
            "completions": ["The sky is blue."],
            "solutions": [{"content": "Accelerating on a straight highway."}],
            "question_type": ["public_lingoqa_action"],
            "expected_range": [(0.0, 0.3)],  # Low similarity
            "desc": "LingoQA-action: Unrelated, low score"
        },
        # Case 8: Batch test (mixed types)
        {
            "completions": ["E", "Between 17 meters and 20 meters", "I am stopping at a red light."],
            "solutions": [
                {"content": "E"},
                {"content": "<answer>Between 17 meters and 20 meters</answer>"},
                {"content": "Stopping for traffic signal."}
            ],
            "question_type": ["public_bdd100k", "public_surds", "public_lingoqa_action"],
            "expected": [1.0, 1.0, None],  # Last one is semantic, check range
            "expected_range": [None, None, (0.4, 1.0)],
            "desc": "Mixed batch test"
        }
    ]

    all_passed = True

    for i, case in enumerate(test_cases, 1):
        print(f"Test {i}: {case['desc']}")
        try:
            rewards = orm(
                completions=case["completions"],
                solutions=case["solutions"],
                question_type=case["question_type"]
            )
            print(f"   Output: {rewards}")

            # Check expected exact values
            expected = case.get("expected")
            if expected:
                for j, (r, exp) in enumerate(zip(rewards, expected)):
                    if exp is not None:
                        assert abs(r - exp) < 1e-5, f"Test {i}.{j}: Expected {exp}, got {r}"

            # Check expected ranges (for semantic cases)
            expected_range = case.get("expected_range")
            if expected_range:
                for j, (r, rng) in enumerate(zip(rewards, expected_range)):
                    if rng is not None:
                        low, high = rng
                        assert low <= r <= high, f"Test {i}.{j}: Reward {r} not in [{low}, {high}]"

            print("   Passed\n")
        except Exception as e:
            print(f"   Failed: {e}\n")
            all_passed = False

    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed.")
        exit(1)


if __name__ == "__main__":
    # Optional: set device info
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    run_tests()
