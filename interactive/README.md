# Interactive Model Usage

This directory contains scripts for interacting with the trained religious text classification model.

## Installation

```bash
pip install -r interactive/requirements.txt
```

Or install directly:
```bash
pip install transformers torch
```

## Scripts

### 1. `predict.py` - Interactive Predictions

Interactive command-line tool for classifying text one at a time.

```bash
python interactive/predict.py
```

**Example:**
```
Enter text: Allah adalah Tuhan yang Maha Esa

📝 Input: 'Allah adalah Tuhan yang Maha Esa'
----------------------------------------
Islam      | ████████████████████ 95.2%
Catholic   | ░░░░░░░░░░░░░░░░░░░░ 3.1%
Protestant | ░░░░░░░░░░░░░░░░░░░░ 1.7%
----------------------------------------
🏆 Prediction: Islam
```

### 2. `predict_batch.py` - Batch Processing

Process multiple texts from a file or command line.

**From file:**
```bash
python interactive/predict_batch.py --file texts.txt
```

**From command line:**
```bash
python interactive/predict_batch.py --text "Text 1" --text "Text 2" --text "Text 3"
```

**Save to CSV:**
```bash
python interactive/predict_batch.py --file texts.txt --output results.csv
```

**Quiet mode (no progress bars):**
```bash
python interactive/predict_batch.py --file texts.txt --quiet
```

## Model

The scripts use the model from Hugging Face:
- **Repository**: `dansachs/indo-religiolect-bert`
- **Model Type**: Sequence Classification
- **Labels**: Islam (0), Catholic (1), Protestant (2)

## Authentication

If the model repository is private, you'll need to login first:

```bash
huggingface-cli login
```

Enter your Hugging Face token when prompted.

## Python API

You can also use the model programmatically:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load model
MODEL_NAME = "dansachs/indo-religiolect-bert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Predict
text = "Allah adalah Tuhan yang Maha Esa"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

probs = F.softmax(logits, dim=1).numpy()[0]
labels = ['Islam', 'Catholic', 'Protestant']
prediction = labels[probs.argmax()]

print(f"Prediction: {prediction}")
print(f"Confidence: {probs.max():.1%}")
```

## Examples

### Example 1: Single Prediction
```bash
python interactive/predict.py
# Enter text when prompted
```

### Example 2: Batch from File
Create `texts.txt`:
```
Allah adalah Tuhan yang Maha Esa
Yesus Kristus adalah Juru Selamat
Roh Kudus membimbing umat
```

Then run:
```bash
python interactive/predict_batch.py --file texts.txt
```

### Example 3: Save Results
```bash
python interactive/predict_batch.py --file texts.txt --output results.csv
```

## Notes

- Model expects Indonesian text
- Maximum sequence length: 128 tokens
- Text is automatically truncated if longer
- Model downloads automatically on first use (~500MB)

