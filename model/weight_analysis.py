import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

OUT_PATH = "model/results"
READ_FILE_PATH = "data/processed/test.jsonl"
os.makedirs(OUT_PATH, exist_ok=True)


def get_attention_weights(model, tokenizer, text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    return outputs.attentions


def visualize_attention(attention_weights, layer_idx=-1, max_tokens=60):
    layer_attn = attention_weights[layer_idx][0].cpu().numpy()
    num_heads = layer_attn.shape[0]
    layer_attn = layer_attn[:, :max_tokens, :max_tokens]

    cols = int(np.sqrt(num_heads))
    if cols * cols < num_heads:
        cols += 1
    rows = (num_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))

    if num_heads == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for head_idx in range(num_heads):
        ax = axes[head_idx]
        sns.heatmap(layer_attn[head_idx], ax=ax, cmap="YlOrRd", cbar=True)
        ax.set_title(f"Head {head_idx}")

    for head_idx in range(num_heads, len(axes)):
        axes[head_idx].axis('off')

    plt.suptitle(f"Attention Weights - Layer {layer_idx} ({num_heads} Heads, {max_tokens} tokens)")
    plt.tight_layout()
    plt.savefig(f"{OUT_PATH}/attention_weights.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/codebert-base",
        num_labels=2
    )
    model = PeftModel.from_pretrained(base_model, "./saved_models/aegis-detect")

    with open(READ_FILE_PATH, "r") as f:
        line = f.readline()
        data = json.loads(line)
        code_snippet = data.get("code")

    attention_weights = get_attention_weights(model, tokenizer, code_snippet)
    visualize_attention(attention_weights, layer_idx=-1, max_tokens=60)


if __name__ == "__main__":
    main()
