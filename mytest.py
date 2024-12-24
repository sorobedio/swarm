from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score
from pathlib import Path
import logging
from torch.cuda import Device


@dataclass
class EvaluationConfig:
    model_path: str
    dataset: str
    gpu_id: int
    batch_size: int = 10
    max_new_tokens: int = 512
    base_model: str = "google/gemma-7b-it"


class ModelEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(f"cuda:{config.gpu_id}")
        self._setup_model()

    def _setup_model(self) -> None:
        """Initialize model and tokenizer."""
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.bfloat16
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logging.error(f"Model initialization failed: {e}")
            raise

    def evaluate_multiple_choice(self, split: str = "dev") -> float:
        """Evaluate model on multiple choice questions."""
        data = self._load_dataset(split)
        prompts = [self._format_mc_prompt(q) for q in data]
        outputs = self._generate_batch(prompts)

        golds = [q["answer"] for q in data]
        preds = [self._parse_mc_answer(q, out) for q, out in zip(data, outputs)]

        return accuracy_score(golds, preds)

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts."""
        outputs = []
        for i in range(0, len(prompts), self.config.batch_size):
            batch = prompts[i:i + self.config.batch_size]
            input_ids = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True
            ).input_ids.to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False
                )

            for j in range(len(batch)):
                decoded = self.tokenizer.decode(
                    output_ids[j][len(input_ids[j]):],
                    skip_special_tokens=True
                ).strip()
                outputs.append(decoded)

        return outputs

    def _load_dataset(self, split: str) -> List[Dict]:
        """Load evaluation dataset."""
        path = Path("data/eval") / f"{self.config.dataset}.json"
        with open(path) as f:
            return json.load(f)[split]

    @staticmethod
    def _format_mc_prompt(question: Dict) -> str:
        """Format multiple choice question prompt."""
        prompt = f"Question: {question['question']}\n"
        for key, value in question['choices'].items():
            prompt += f"{key}: {value}\n"
        prompt += "The answer is"
        return prompt

    @staticmethod
    def _parse_mc_answer(question: Dict, output: str) -> str:
        """Parse model output for multiple choice answer."""
        for key in question["choices"]:
            if key in output[:5] or key in output[-5:]:
                return key
            if question["choices"][key].lower() in output.lower():
                return key
        return "Z"  # Invalid/no answer


def main():
    config = EvaluationConfig(
        model_path="google/gemma-7b-it",
        dataset="mmlu",
        gpu_id=0
    )

    evaluator = ModelEvaluator(config)
    result = evaluator.evaluate_multiple_choice()
    print(f"Evaluation result: {result:.4f}")


if __name__ == "__main__":
    main()