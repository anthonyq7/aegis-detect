# Aegis: AI vs. Human Python Code Classifier

## Overview
Aegis is a fine-tuned CodeBERT model designed to classify AI-generated and human Python code. While CodeBERT contains 125 million parameters, Aegis was efficiently trained locally using LoRA (Low-Rank Adaptation), updating only a subset of the original parameters.

This project investigates classifying code based on semantic differences. Consequently, the dataset (20K Python samples: 10K AI + 10K Human) was aggressively cleaned to ensure standard formatting and the removal of comments and docstrings. A confidence threshold of 0.7 was established to flag samples as AI-generated only when strong evidence exists. Aegis is not a definitive judge; predictions can be imperfect, particularly in tasks where semantic convergence between humans and AI is observed (e.g., LeetCode solutions).

## Installation

```bash
pip install aegis-detect
```

### CLI Usage

**Supported commands**:
```bash
# Predicting using a file
aegis --file path/to/code.py

# Predicting using text
aegis --text "def add(a, b):\n    return a + b"

# JSON output
aegis --file path/to/code.py --json > result.json

# Setting a threshold for AI classification 
aegis --file path/to/code.py --threshold 0.7

# Help
aegis --help

# Uninstall
aegis-cleanup 
pip uninstall aegis-detect
```

**Notes**:
- On first run, the model adapter is downloaded from the Hugging Face repo [anthonyq7/aegis](https://huggingface.co/anthonyq7/aegis) and cached under `~/.aegis/models`.
- Internet access is required on the first run; subsequent runs use local cache. 
- The CLI prints the predicted label and probabilities for human and AI.

## Key Results

### Model Performance
- **Accuracy**: 85.10%
- **Precision**: 83.37%
- **Recall**: 87.70%
- **F1-Score**: 85.48%

### Confusion Matrix
![Alt text](model/results/confusion_matrix.png)

### Attention Heatmap
![Alt text](model/results/attention_weights.png)

## Contact
**Email**: a.j.qin@wustl.edu

**Email**: ethanqin@bu.edu

## License
This project is licensed under the [MIT License](LICENSE).
