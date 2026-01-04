#!/usr/bin/env python3
"""
Inference script for religious text classification model.

Can load models from:
- Hugging Face Hub (e.g., "username/model-name")
- Local directory (e.g., "./models/trained/model_name")
- Google Drive path (if mounted in Colab)

Usage:
    python inference.py --text "Your text here"
    python inference.py --model "username/model-name" --text "Your text here"
    python inference.py --model "./models/trained/model_name" --text "Your text here"
    python inference.py --file input.txt
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Union
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import numpy as np

# Default model from Hugging Face
DEFAULT_MODEL = "dansachs/indo-religiolect-bert-v2"

# Label mapping (default - will be loaded from model if available)
DEFAULT_LABEL_MAP = {0: 'Islam', 1: 'Catholic', 2: 'Protestant'}


class ReligiousTextClassifier:
    """Classifier for Indonesian religious texts."""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to model (Hugging Face Hub ID or local path)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"📥 Loading model from: {model_path}")
        print(f"🖥️  Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Load label mapping if available
        self.label_map = self._load_label_map(model_path)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Labels: {list(self.label_map.values())}")
    
    def _load_label_map(self, model_path: str) -> Dict[int, str]:
        """Load label mapping from model directory or Hugging Face Hub."""
        # Try to load from local model directory
        model_dir = Path(model_path)
        if model_dir.exists() and model_dir.is_dir():
            label_map_file = model_dir / "label_map.json"
            if label_map_file.exists():
                with open(label_map_file, 'r') as f:
                    label_map_reverse = json.load(f)
                    # Convert from {"Islam": 0, ...} to {0: "Islam", ...}
                    return {v: k for k, v in label_map_reverse.items()}
        
        # Try to load from Hugging Face Hub
        if "/" in model_path and not Path(model_path).exists():
            try:
                label_map_file = hf_hub_download(
                    repo_id=model_path,
                    filename="label_map.json",
                    repo_type="model"
                )
                with open(label_map_file, 'r') as f:
                    label_map_reverse = json.load(f)
                    # Convert from {"Islam": 0, ...} to {0: "Islam", ...}
                    return {v: k for k, v in label_map_reverse.items()}
            except Exception:
                # If file doesn't exist on Hub, fall through to default
                pass
        
        # Fallback to default
        return DEFAULT_LABEL_MAP
    
    def predict(self, text: Union[str, List[str]], return_probs: bool = False) -> Union[Dict, List[Dict]]:
        """
        Predict religion for given text(s).
        
        Args:
            text: Single text string or list of texts
            return_probs: If True, return probabilities for all classes
        
        Returns:
            If single text: Dict with 'label', 'confidence', and optionally 'probabilities'
            If list: List of such dicts
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            predictions = np.argmax(probs, axis=-1)
            confidences = np.max(probs, axis=-1)
        
        # Format results
        results = []
        for i, (pred, conf, prob) in enumerate(zip(predictions, confidences, probs)):
            result = {
                'label': self.label_map[pred],
                'confidence': float(conf),
                'text': texts[i]
            }
            
            if return_probs:
                result['probabilities'] = {
                    self.label_map[j]: float(prob[j])
                    for j in range(len(self.label_map))
                }
            
            results.append(result)
        
        return results[0] if is_single else results
    
    def predict_batch(self, texts: List[str], batch_size: int = 32, return_probs: bool = False) -> List[Dict]:
        """
        Predict for a batch of texts (handles large batches efficiently).
        
        Args:
            texts: List of texts to classify
            batch_size: Number of texts to process at once
            return_probs: If True, return probabilities for all classes
        
        Returns:
            List of prediction dicts
        """
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.predict(batch, return_probs=return_probs)
            all_results.extend(batch_results)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"   Processed {min(i + batch_size, len(texts))}/{len(texts)} texts...")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Classify Indonesian religious texts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single text from command line (uses default Hugging Face model)
  python inference.py --text "Teks dalam bahasa Indonesia"
  
  # Single text with specific model
  python inference.py --model "username/model-name" --text "Teks dalam bahasa Indonesia"
  
  # Multiple texts from file
  python inference.py --file texts.txt
  
  # With probabilities
  python inference.py --text "Teks" --probs
  
  # Use CPU instead of GPU
  python inference.py --text "Teks" --device cpu
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        help=f'Model path (Hugging Face Hub ID or local path, default: {DEFAULT_MODEL})'
    )
    
    parser.add_argument(
        '--text',
        type=str,
        help='Single text to classify'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='File with texts (one per line)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default=None,
        help='Device to use (default: auto-detect)'
    )
    
    parser.add_argument(
        '--probs',
        action='store_true',
        help='Show probabilities for all classes'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for file processing (default: 32)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for results (JSON format)'
    )
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = ReligiousTextClassifier(args.model, device=args.device)
    
    # Get texts
    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"📂 Loaded {len(texts)} texts from {args.file}")
    else:
        parser.error("Either --text or --file must be provided")
    
    # Predict
    print(f"\n🔍 Classifying {len(texts)} text(s)...")
    if len(texts) == 1:
        result = classifier.predict(texts[0], return_probs=args.probs)
        print(f"\n📊 Result:")
        print(f"   Text: {result['text']}")
        print(f"   Label: {result['label']}")
        print(f"   Confidence: {result['confidence']:.4f}")
        if args.probs:
            print(f"   Probabilities:")
            for label, prob in result['probabilities'].items():
                print(f"      {label}: {prob:.4f}")
        
        results = [result]
    else:
        results = classifier.predict_batch(texts, batch_size=args.batch_size, return_probs=args.probs)
        print(f"\n📊 Results:")
        for i, result in enumerate(results[:10], 1):  # Show first 10
            print(f"   {i}. {result['label']} ({result['confidence']:.4f})")
        if len(results) > 10:
            print(f"   ... and {len(results) - 10} more")
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {args.output}")
    
    return results


if __name__ == "__main__":
    main()

