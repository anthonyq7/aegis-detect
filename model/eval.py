import json
import os

import matplotlib.pyplot as plt
import torch
from dataset import CodeDataset
from peft import PeftModel
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import get_device

IN_PATH = "data/processed"
OUT_PATH = "model/results"
THRESHOLD = 0.7

def eval():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/codebert-base",
        num_labels=2
    )

    os.makedirs(OUT_PATH, exist_ok=True)

    model = PeftModel.from_pretrained(base_model, "./saved_models/aegis-detect")
    test_dataset = CodeDataset(f"{IN_PATH}/test.jsonl", tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = get_device()
    model = model.to(device)
    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            batch_predictions = (probs[:, 1] >= THRESHOLD).int().cpu().numpy()

            predictions.extend(batch_predictions.tolist())
            true_labels.extend(labels.cpu().numpy().tolist())

    cm = confusion_matrix(true_labels, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Human", "AI-Generated"]
    )

    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(f"{OUT_PATH}/confusion_matrix.png", dpi=300)
    plt.close()

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    results = {
        "model_name": "Aegis",
        "metrics": {
            "accuracy": f"{accuracy:.4f}",
            "precision": f"{precision:.4f}",
            "recall": f"{recall:.4f}",
            "f1": f"{f1:.4f}",
        },
        "confusion_matrix": {
            "values": cm.tolist(),
            "labels": ["Human", "AI-Generated"],
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1])
        }
    }

    with open(f"{OUT_PATH}/model_results.json", "w") as file:
        file.write(json.dumps(results, indent=4))

    print(f"Results saved to {OUT_PATH}/model_results.json")

if __name__ == "__main__":
    eval()
