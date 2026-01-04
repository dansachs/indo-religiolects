# Indonesian Religious Language Model (Religiolect Classifier)

An NLP project to classify Indonesian text into three distinct religious denominations: Islam, Catholicism, and Protestantism.

This model uses Transfer Learning (fine-tuning IndoBERT) to identify the unique "religiolects" (religious dialects) used by different faith communities in Indonesia. It successfully distinguishes between groups with high accuracy, even navigating the shared vocabulary between Catholic and Protestant discourse.

## 📂 Data & Model Access

Due to GitHub file size limits, the dataset and trained model weights are hosted externally.

### 1. The Dataset

The corpus consists of ~3 million clean sentences scraped from 100+ authoritative religious websites (NU Online, Mirifica, PGI, KAS, and many others).

**Format:** CSV (Columns: Label, Denomination, Location, Date, Title, Sentence_Unit, Link)  
**Size:** ~3 million sentences  
**Sources:** 30 Catholic sites, 27 Islamic sites, 44 Protestant sites

🔗 *Dataset hosted externally (not in this repository)*

### 2. The Model

The fine-tuned BERT model is available on the Hugging Face Hub for easy inference.

**Model Repository:** `dansachs/indo-religiolect-bert`  
**Base Model:** `indolem/indobert-base-uncased`  
**Training:** Fine-tuned with balanced undersampling strategy  
**Classes:** Islam (0), Catholic (1), Protestant (2)

🔗 [View Model on Hugging Face](https://huggingface.co/dansachs/indo-religiolect-bert)

## 🛠️ Installation

Clone the repository:
```bash
git clone https://github.com/dansachs/indo-religious-lang/
cd indonesian-religious-model
```

Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

For interactive predictions, install the interactive requirements:
```bash
pip install -r interactive/requirements.txt
```

## 🚀 Quick Start

### Using the Pre-trained Model (Inference)

**Interactive mode:**
```bash
python interactive/predict.py
```

**Batch processing:**
```bash
python interactive/predict_batch.py --file texts.txt --output results.csv
```

### Training Your Own Model

**Local training:**
```bash
python training/train_model.py
```

**Google Colab:**
- Upload `training/train_model_colab.ipynb` to Google Colab
- Follow the notebook instructions

### Data Collection (Building the Corpus)

Run the crawler to collect data from religious websites:
```bash
python run_crawler.py
```

See the [full documentation](docs/) for detailed usage instructions.

## 📁 Project Structure

```
religiolect_model_V2/
├── src/                    # Core crawler source code
│   ├── crawler.py         # Async crawler with rate limiting
│   ├── nlp_processor.py   # Text extraction & cleaning
│   ├── dashboard.py       # Live terminal dashboard
│   └── run_pipeline.py    # Pipeline orchestration
│
├── training/               # Model training scripts
│   ├── train_model.py              # Local training script
│   ├── train_model_colab.ipynb    # IndoBERT Colab notebook
│   └── inference.py                # Inference script
│
├── interactive/            # Interactive model usage
│   ├── predict.py          # Interactive predictions
│   ├── predict_batch.py    # Batch processing
│   └── README.md           # Usage guide
│
├── data/                   # Data files (generated)
│   ├── scraped/           # Individual religion CSV files
│   └── combined/          # Combined datasets
│
├── configs/                # Configuration files
│   └── seeds.json         # Target URLs for crawling
│
└── docs/                   # Documentation
    ├── PROJECT_STRUCTURE.md
    ├── INFERENCE_GUIDE.md
    └── COLAB_SETUP.md
```

## 🔧 Usage Examples

### Python API

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

## 📊 Model Performance

The model is fine-tuned on a balanced dataset using undersampling to ensure equal representation across all three religious classes. Training uses the IndoBERT base model, which is specifically pre-trained on Indonesian text, providing superior performance for Indonesian language tasks.

## 🔄 Data Collection Pipeline

This repository includes a complete data collection pipeline:

- **Ethical Crawling**: 1 request per domain, adaptive backoff on 429/503 errors
- **Resumable**: Stop and resume crawling at any time
- **Live Dashboard**: Real-time progress tracking
- **Smart Filtering**: Language detection, noise removal, scripture reference cleaning

See the [main README documentation](docs/) for detailed information on data collection.

## 📜 License

For academic research purposes.

## 🙏 Acknowledgments

- Base model: [IndoBERT by IndoLEM](https://huggingface.co/indolem/indobert-base-uncased)
- Built with [Hugging Face Transformers](https://huggingface.co/transformers/)
