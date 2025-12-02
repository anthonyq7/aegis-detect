"""Core logic for the CLI interface"""

from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Predictor:

    def __init__(self, threshold: Optional[float] = None, model_name="anthonyq7/aegis"):

        if threshold:
            if not (0.0 <= threshold <= 1.0):
                raise ValueError("Threshold must be a float between 0 and 1")

        if threshold is None:
            self.threshold = 0.7
        else:
            self.threshold = float(threshold)

        self.model_path = self._get_or_download_model(model_name)
        self._load_model()

    def _get_or_download_model(self, model_name):
        cached_dir = Path.home() / ".aegis" / "models"
        cached_dir.mkdir(parents=True, exist_ok=True)

        if not any(cached_dir.iterdir()):
            try:
                snapshot_download(
                    repo_id=model_name,
                    local_dir=cached_dir
                )
                print("\u2713 Model downloaded successfully!")
            except Exception as e:
                print(f"\u2717 Error downloading mode: {e}")
                print(f"Make sure the model exists at huggingface.co/{model_name}")
                raise

        return cached_dir

    def _load_model(self):

        print("Loading model...")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        base_model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.model = PeftModel.from_pretrained(base_model, str(self.model_path))
        self.model.eval()

        print("\u2713 Model loaded!")

    def predict(self, code):

        if not code or not code.strip():
            raise ValueError("Code cannot be empty")

        inputs = self.tokenizer(
            code,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy().tolist()[0]

        human_prob = f"{float(probs[0]):.4f}"
        ai_prob = f"{float(probs[1]):.4f}"

        return {
            "human": human_prob,
            "ai": ai_prob,
            "prediction": "ai-generated" if float(ai_prob) >= self.threshold else "human"
        }










