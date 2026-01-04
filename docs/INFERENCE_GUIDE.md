# Inference Guide

This guide shows how to use the trained model to classify new Indonesian religious texts.

## Quick Start

### 1. Load Model from Hugging Face Hub

If you've uploaded your model to Hugging Face:

```bash
python training/inference.py \
  --model "your-username/religious-text-classifier" \
  --text "Teks dalam bahasa Indonesia yang ingin diklasifikasi"
```

### 2. Load Model from Local Directory

If you have the model saved locally:

```bash
python training/inference.py \
  --model "./models/trained/indobert-base-uncased_20241218_120000" \
  --text "Teks dalam bahasa Indonesia"
```

### 3. Load Model from Google Drive (Colab)

If you saved the model to Google Drive in Colab:

```bash
python training/inference.py \
  --model "/content/drive/MyDrive/Indo_Religiolect/model_final" \
  --text "Teks dalam bahasa Indonesia"
```

## Usage Examples

### Single Text Classification

```bash
python training/inference.py \
  --model "username/model-name" \
  --text "Allah adalah Tuhan yang Maha Esa"
```

**Output:**
```
📥 Loading model from: username/model-name
🖥️  Using device: cuda
✅ Model loaded successfully!
   Labels: ['Islam', 'Catholic', 'Protestant']

🔍 Classifying 1 text(s)...

📊 Result:
   Text: Allah adalah Tuhan yang Maha Esa
   Label: Islam
   Confidence: 0.9876
```

### With Probabilities

See confidence scores for all classes:

```bash
python training/inference.py \
  --model "username/model-name" \
  --text "Teks" \
  --probs
```

**Output:**
```
📊 Result:
   Text: Teks
   Label: Islam
   Confidence: 0.9876
   Probabilities:
      Islam: 0.9876
      Catholic: 0.0089
      Protestant: 0.0035
```

### Batch Processing from File

Classify multiple texts from a file (one text per line):

```bash
python training/inference.py \
  --model "username/model-name" \
  --file texts.txt \
  --batch-size 64
```

**texts.txt:**
```
Allah adalah Tuhan yang Maha Esa
Yesus Kristus adalah Juru Selamat
Roh Kudus membimbing umat
```

### Save Results to JSON

```bash
python training/inference.py \
  --model "username/model-name" \
  --file texts.txt \
  --output results.json
```

**results.json:**
```json
[
  {
    "label": "Islam",
    "confidence": 0.9876,
    "text": "Allah adalah Tuhan yang Maha Esa",
    "probabilities": {
      "Islam": 0.9876,
      "Catholic": 0.0089,
      "Protestant": 0.0035
    }
  },
  ...
]
```

## Python API

You can also use the classifier programmatically:

```python
from training.inference import ReligiousTextClassifier

# Initialize
classifier = ReligiousTextClassifier("username/model-name")

# Single prediction
result = classifier.predict("Teks dalam bahasa Indonesia")
print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']}")

# With probabilities
result = classifier.predict("Teks", return_probs=True)
print(result['probabilities'])

# Batch prediction
texts = ["Teks 1", "Teks 2", "Teks 3"]
results = classifier.predict_batch(texts, batch_size=32)
for result in results:
    print(f"{result['text']} -> {result['label']} ({result['confidence']:.2f})")
```

## Uploading Model to Hugging Face

To use your model from Hugging Face Hub:

1. **Install Hugging Face Hub:**
   ```bash
   pip install huggingface_hub
   ```

2. **Login:**
   ```bash
   huggingface-cli login
   ```

3. **Upload model:**
   ```python
   from huggingface_hub import HfApi
   
   api = HfApi()
   api.upload_folder(
       folder_path="./models/trained/your_model",
       repo_id="your-username/religious-text-classifier",
       repo_type="model"
   )
   ```

4. **Use it:**
   ```bash
   python training/inference.py \
     --model "your-username/religious-text-classifier" \
     --text "Your text"
   ```

## Device Selection

Force CPU usage (useful if GPU has memory issues):

```bash
python training/inference.py \
  --model "username/model-name" \
  --text "Teks" \
  --device cpu
```

## Notes

- The model expects Indonesian text
- Maximum sequence length: 128 tokens (longer texts are truncated)
- Batch processing is more efficient for multiple texts
- Model automatically detects GPU/CPU availability

