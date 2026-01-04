#!/usr/bin/env python3
"""
Train a religious text classification model using transformers.

Supports multiple Indonesian language models:
- indolem/indobert-base-uncased (recommended for Indonesian)
- bert-base-multilingual-cased (multilingual BERT)
- xlm-roberta-base (multilingual RoBERTa)
- distilbert-base-multilingual-cased (lighter, faster)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
import torch
import os
from datetime import datetime

# Disable WandB (optional logging service)
os.environ["WANDB_DISABLED"] = "true"

# ==========================================
# CONFIGURATION
# ==========================================

# Model options (uncomment one to use)
MODEL_NAME = "indolem/indobert-base-uncased"  # Recommended for Indonesian
# MODEL_NAME = "bert-base-multilingual-cased"
# MODEL_NAME = "xlm-roberta-base"
# MODEL_NAME = "distilbert-base-multilingual-cased"

# File paths
BASE_DIR = Path(__file__).parent.parent.resolve()  # Go up one level from training/ to project root
COMBINED_DATA_DIR = BASE_DIR / "data" / "combined"

# Auto-detect latest combined CSV file
CSV_FILES = sorted(COMBINED_DATA_DIR.glob("religious_corpus_*.csv"), reverse=True)
if CSV_FILES:
    FILE_PATH = CSV_FILES[0]
    print(f"📂 Using latest CSV: {FILE_PATH.name}")
else:
    # Fallback: specify manually if needed
    FILE_PATH = COMBINED_DATA_DIR / "religious_corpus_20251218_115315.csv"

# Output directory for trained model
MODEL_OUTPUT_DIR = BASE_DIR / "models" / "trained" / f"{MODEL_NAME.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 32  # Reduce to 16 if you get Out of Memory errors
EVAL_BATCH_SIZE = 64
MAX_LENGTH = 128  # Maximum sequence length for tokenization
TEST_SIZE = 0.1  # 10% for testing

# ==========================================
# 1. LOAD & BALANCE DATA
# ==========================================

print("\n" + "="*60)
print("📊 LOADING & PREPARING DATA")
print("="*60)

print(f"\nLoading dataset from: {FILE_PATH}")
if not FILE_PATH.exists():
    raise FileNotFoundError(f"Dataset file not found: {FILE_PATH}")

df = pd.read_csv(FILE_PATH)
print(f"✅ Loaded {len(df):,} rows")

# Map text labels to integers
label_map = {'Islam': 0, 'Catholic': 1, 'Protestant': 2}
df['label'] = df['Label'].map(label_map)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

print(f"\n📈 Original class distribution:")
print(df['Label'].value_counts())

# --- UNDERSAMPLING STRATEGY ---
# Find the size of the smallest class
min_class_size = df['label'].value_counts().min()
print(f"\n⚖️  Balancing data (smallest class: {min_class_size:,} samples)")

# Sample that amount from each group
df_balanced = df.groupby('label').apply(
    lambda x: x.sample(min_class_size, random_state=42)
).reset_index(drop=True)

print("\n✅ Balanced class distribution:")
print(df_balanced['Label'].value_counts())

# Split Data (90% Train, 10% Test)
print(f"\n📊 Splitting data ({int((1-TEST_SIZE)*100)}% train, {int(TEST_SIZE*100)}% test)...")
train_df, test_df = train_test_split(
    df_balanced, 
    test_size=TEST_SIZE, 
    stratify=df_balanced['label'], 
    random_state=42
)

print(f"   Train: {len(train_df):,} samples")
print(f"   Test: {len(test_df):,} samples")

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# ==========================================
# 2. TOKENIZATION
# ==========================================

print("\n" + "="*60)
print("🔤 TOKENIZATION")
print("="*60)

print(f"\n📥 Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    """Tokenize sentences with padding and truncation."""
    return tokenizer(
        examples["Sentence_Unit"], 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_LENGTH
    )

print(f"\n🔄 Tokenizing data (max_length={MAX_LENGTH})...")
print("   This may take a few minutes...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)
print("✅ Tokenization complete!")

# ==========================================
# 3. METRICS
# ==========================================

def compute_metrics(pred):
    """Compute classification metrics."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc, 
        'f1': f1, 
        'precision': precision, 
        'recall': recall
    }

# ==========================================
# 4. MODEL SETUP
# ==========================================

print("\n" + "="*60)
print("🤖 MODEL SETUP")
print("="*60)

print(f"\n📥 Loading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=3
)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

model.to(device)

# ==========================================
# 5. TRAINING ARGUMENTS
# ==========================================

print("\n" + "="*60)
print("⚙️  TRAINING CONFIGURATION")
print("="*60)

training_args = TrainingArguments(
    output_dir=str(MODEL_OUTPUT_DIR / "checkpoints"),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=2e-5,  # Standard learning rate for BERT fine-tuning
    logging_dir=str(MODEL_OUTPUT_DIR / "logs"),
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,  # Only keep the best model to save space
    load_best_model_at_end=True,
    metric_for_best_model="f1",  # Use F1 score to select best model
    greater_is_better=True,
    fp16=torch.cuda.is_available(),  # Mixed precision (faster on GPU)
    report_to="none",  # Disable all logging services
)

print(f"\n📋 Training configuration:")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Learning rate: {training_args.learning_rate}")
print(f"   Output directory: {MODEL_OUTPUT_DIR}")

# ==========================================
# 6. TRAINER SETUP
# ==========================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

# ==========================================
# 7. TRAINING
# ==========================================

print("\n" + "="*60)
print("🚀 STARTING TRAINING")
print("="*60)
print(f"\n⏱️  This may take a while...")
print(f"   Model: {MODEL_NAME}")
print(f"   Training samples: {len(train_df):,}")
print(f"   Test samples: {len(test_df):,}")

trainer.train()

# ==========================================
# 8. SAVE MODEL
# ==========================================

print("\n" + "="*60)
print("💾 SAVING MODEL")
print("="*60)

print(f"\n💾 Saving model to: {MODEL_OUTPUT_DIR}")
# Save the best model (trainer.load_best_model_at_end=True ensures this is the best)
model.save_pretrained(str(MODEL_OUTPUT_DIR))
tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))

# Save label mapping
import json
with open(MODEL_OUTPUT_DIR / "label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)

# Save training info
training_info = {
    "model_name": MODEL_NAME,
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "eval_batch_size": EVAL_BATCH_SIZE,
    "max_length": MAX_LENGTH,
    "train_samples": len(train_df),
    "test_samples": len(test_df),
    "label_map": label_map
}
with open(MODEL_OUTPUT_DIR / "training_info.json", "w") as f:
    json.dump(training_info, f, indent=2)

print("✅ Model saved successfully!")
print(f"   📁 Model: {MODEL_OUTPUT_DIR}")
print(f"   📁 Tokenizer: {MODEL_OUTPUT_DIR}")
print(f"   📁 Label map: {MODEL_OUTPUT_DIR / 'label_map.json'}")
print(f"   📁 Training info: {MODEL_OUTPUT_DIR / 'training_info.json'}")

# ==========================================
# 9. EVALUATION & VISUALIZATION
# ==========================================

print("\n" + "="*60)
print("📊 EVALUATION")
print("="*60)

print("\n🔍 Generating predictions on test set...")
predictions = trainer.predict(tokenized_test)
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

# Calculate metrics
accuracy = accuracy_score(labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(
    labels, preds, average='weighted'
)

print(f"\n📈 Final Metrics:")
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   F1 Score:  {f1:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")

# Confusion Matrix
print("\n📊 Generating confusion matrix...")
cm = confusion_matrix(labels, preds)
label_names = list(label_map.keys())

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=label_names,
    yticklabels=label_names
)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title(f'Confusion Matrix: Religious Style Detection\nModel: {MODEL_NAME.split("/")[-1]}')
plt.tight_layout()

# Save confusion matrix
cm_path = MODEL_OUTPUT_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"✅ Confusion matrix saved: {cm_path}")

# Show plot (if running interactively)
try:
    plt.show()
except:
    print("   (Plot display not available in this environment)")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print(f"\n📁 Model saved to: {MODEL_OUTPUT_DIR}")
print(f"📊 Confusion matrix: {cm_path}")
print("\n💡 To use this model:")
print(f"   from transformers import AutoTokenizer, AutoModelForSequenceClassification")
print(f"   tokenizer = AutoTokenizer.from_pretrained('{MODEL_OUTPUT_DIR}')")
print(f"   model = AutoModelForSequenceClassification.from_pretrained('{MODEL_OUTPUT_DIR}')")

