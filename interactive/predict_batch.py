#!/usr/bin/env python3
"""
Batch prediction script for religious text classification.

Processes multiple texts from a file or command line arguments.

Usage:
    python interactive/predict_batch.py --file texts.txt
    python interactive/predict_batch.py --text "Text 1" --text "Text 2"
"""

import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
MODEL_NAME = "dansachs/indo-religiolect-bert"
LABELS = ['Islam', 'Catholic', 'Protestant']

def load_model():
    """Load model and tokenizer."""
    print(f"⬇️ Loading model: {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        print("✅ Model loaded successfully!\n")
        return tokenizer, model
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Tip: If the repo is Private, you need to log in with `huggingface-cli login` first.")
        raise

def predict(text, tokenizer, model):
    """Predict religion for a single text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    probs = F.softmax(logits, dim=1).numpy()[0]
    prediction = LABELS[probs.argmax()]
    confidence = probs.max()
    
    return {
        'text': text,
        'prediction': prediction,
        'confidence': confidence,
        'probabilities': {label: float(prob) for label, prob in zip(LABELS, probs)}
    }

def main():
    parser = argparse.ArgumentParser(description="Batch religious text classification")
    parser.add_argument('--file', type=str, help='File with texts (one per line)')
    parser.add_argument('--text', action='append', help='Text to classify (can use multiple times)')
    parser.add_argument('--output', type=str, help='Output CSV file')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode (no progress bars)')
    
    args = parser.parse_args()
    
    # Load model
    tokenizer, model = load_model()
    
    # Get texts
    texts = []
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ File not found: {args.file}")
            return
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"📂 Loaded {len(texts)} texts from {args.file}\n")
    elif args.text:
        texts = args.text
    else:
        parser.error("Either --file or --text must be provided")
    
    # Predict
    results = []
    for i, text in enumerate(texts, 1):
        if not args.quiet:
            print(f"Processing {i}/{len(texts)}...", end='\r')
        result = predict(text, tokenizer, model)
        results.append(result)
    
    if not args.quiet:
        print()  # New line after progress
    
    # Display results
    print("\n" + "="*60)
    print("📊 Results")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['text'][:60]}{'...' if len(result['text']) > 60 else ''}")
        print(f"   🏆 Prediction: {result['prediction']} ({result['confidence']:.1%})")
        if not args.quiet:
            for label, prob in result['probabilities'].items():
                bar_len = int(prob * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(f"      {label:<10} | {bar} {prob:.1%}")
    
    # Save to CSV if requested
    if args.output:
        import csv
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'prediction', 'confidence', 'islam_prob', 'catholic_prob', 'protestant_prob'])
            writer.writeheader()
            for result in results:
                writer.writerow({
                    'text': result['text'],
                    'prediction': result['prediction'],
                    'confidence': f"{result['confidence']:.4f}",
                    'islam_prob': f"{result['probabilities']['Islam']:.4f}",
                    'catholic_prob': f"{result['probabilities']['Catholic']:.4f}",
                    'protestant_prob': f"{result['probabilities']['Protestant']:.4f}",
                })
        print(f"\n💾 Results saved to: {args.output}")

if __name__ == "__main__":
    main()

