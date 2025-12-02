import os
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataset import CodeDataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from utils import get_device

IN_PATH = "data/processed"
THRESHOLD = 0.7  

os.makedirs(IN_PATH, exist_ok=True)

def compute_metrics(eval_pred):

    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    predictions = (probs[:, 1] >= THRESHOLD).astype(int)
  
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "f1": f1_score(labels, predictions)
    }

def train():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/codebert-base",
        num_labels=2
    )

    device = get_device()
    print(f"Using {device} for training.")

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["query", "value", "key", "dense"],
        modules_to_save=["classifier", "score"]
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    train_dataset = CodeDataset(f"{IN_PATH}/train.jsonl", tokenizer)
    val_dataset = CodeDataset(f"{IN_PATH}/val.jsonl", tokenizer)

    training_args = TrainingArguments(
        output_dir="./saved_models/aegis",
        gradient_checkpointing=False,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        bf16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=True,
        learning_rate=3e-4,
        num_train_epochs=5,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine_with_restarts",
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        neftune_noise_alpha=5,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print("\nStarting training")
    trainer.train()

    print("\nSaving model")
    model.save_pretrained("./saved_models/aegis-detect")
    tokenizer.save_pretrained("./saved_models/aegis-detect")
    print("Done")


if __name__ == "__main__":
    train()