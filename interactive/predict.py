#!/usr/bin/env python3
"""
Interactive religious text classifier.

Loads the trained model from Hugging Face and allows interactive predictions.

Usage:
    python interactive/predict.py
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np

# --- CONFIGURATION ---
# Your specific model ID from Hugging Face
MODEL_NAME = "dansachs/indo-religiolect-bert-v2" 

print(f"⬇️ Loading model: {MODEL_NAME}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Tip: If the repo is Private, you need to log in with `huggingface-cli login` first.")
    exit()

# --- PREDICTION FUNCTION ---
def predict_religion(text):
    """Predict religion for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    probs = F.softmax(logits, dim=1).numpy()[0]
    labels = ['Islam', 'Catholic', 'Protestant']
    
    print(f"\n📝 Input: '{text}'")
    print("-" * 40)
    for label, prob in zip(labels, probs):
        bar_len = int(prob * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"{label:<10} | {bar} {prob:.1%}")
    
    winner = labels[probs.argmax()]
    print("-" * 40)
    print(f"🏆 Prediction: {winner}")

# --- INTERACTIVE MODE ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🔮 Religious Text Classifier")
    print("="*50)
    print("\nType Indonesian text to classify, or 'q' to quit.\n")
    
    while True:
        input_sentence = input("Enter text: ").strip()
        
        if input_sentence.lower() in ['q', 'quit', 'exit']:
            print("👋 Goodbye!")
            break
        
        if not input_sentence:
            print("⚠️  Please enter some text.")
            continue
        
        try:
            predict_religion(input_sentence)
            print()  # Empty line for readability
        except Exception as e:
            print(f"❌ Error: {e}\n")

