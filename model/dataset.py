from typing import Any, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class CodeDataset(Dataset):
    
    def __init__(self, filepath: str, tokenizer: AutoTokenizer, max_length: int = 512) -> None:
        self.data = pd.read_json(filepath, lines=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Loaded {len(self.data)} samples from {filepath}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.data.iloc[index]
        encoding = self.tokenizer(
            row["code"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(row["label"], dtype=torch.long)
        }
