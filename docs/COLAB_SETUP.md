# Google Colab Setup Guide

## Quick Start

1. **Upload your CSV file to Google Drive**
   - Upload `religious_corpus_*.csv` to your Google Drive
   - Note the path (e.g., `/content/drive/MyDrive/religious_corpus_20251218_115315.csv`)

2. **Open the notebook in Colab**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Click "File" → "Upload notebook"
   - Upload `train_model_colab.ipynb`

3. **Update the configuration**
   - In cell 3, update `FILE_PATH` with your CSV file path
   - Choose your model (uncomment one in cell 3)
   - Optionally adjust batch size if you get memory errors

4. **Run all cells**
   - Click "Runtime" → "Run all"
   - Or run cells one by one with Shift+Enter

## Model Options

The notebook supports these models (uncomment one in cell 3):

- **`indolem/indobert-base-uncased`** ⭐ Recommended for Indonesian
- `bert-base-multilingual-cased` - Multilingual BERT
- `xlm-roberta-base` - Multilingual RoBERTa
- `distilbert-base-multilingual-cased` - Lighter, faster

## Tips

- **GPU**: Colab provides free GPU (T4). Make sure "Runtime" → "Change runtime type" → "GPU" is selected
- **Memory errors**: Reduce `BATCH_SIZE` to 16 if you get Out of Memory errors
- **Save path**: Models are saved to Google Drive automatically
- **Training time**: Expect 30-60 minutes depending on model and data size

## Output

After training, you'll get:
- Trained model saved to Google Drive
- Confusion matrix visualization
- Evaluation metrics (accuracy, F1, precision, recall)
- Label mapping file (`label_map.json`)

## Using the Trained Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model_path = "/content/drive/MyDrive/religiolect_models/indobert-base-uncased_20251218_120000"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Predict
text = "Banyak orang yang bertanya-tanya..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=-1).item()

label_map = {0: 'Islam', 1: 'Catholic', 2: 'Protestant'}
print(f"Prediction: {label_map[prediction]}")
```

