#!/usr/bin/env python3
"""
Upload the Indonesian Religious Corpus dataset to Hugging Face.

Usage:
    python scripts/upload_dataset_to_hf.py
"""

import os
import sys
from pathlib import Path

# Set environment variable before importing datasets to avoid pyarrow compatibility issues
os.environ['HF_DATASETS_USE_CONTENT_DEFINED_CHUNKING'] = 'false'

from datasets import Dataset, DatasetDict
from huggingface_hub import login
import pandas as pd

# Configuration
DATASET_PATH = Path(__file__).parent.parent / "data" / "combined" / "religious_corpus_20251218_162005.csv"
REPO_ID = "dansachs/indonesian-religious-corpus"

def main():
    print("=" * 60)
    print("📤 Uploading Indonesian Religious Corpus to Hugging Face")
    print("=" * 60)
    
    # Check if dataset file exists
    if not DATASET_PATH.exists():
        print(f"❌ Dataset file not found: {DATASET_PATH}")
        sys.exit(1)
    
    print(f"\n📂 Dataset file: {DATASET_PATH}")
    print(f"📦 Repository: {REPO_ID}")
    
    # Login to Hugging Face
    print("\n🔐 Logging in to Hugging Face...")
    try:
        login()
        print("✅ Logged in successfully!")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        print("💡 Make sure you have a Hugging Face token set up.")
        print("   Run: huggingface-cli login")
        sys.exit(1)
    
    # Load dataset
    print("\n📖 Loading dataset...")
    print("   This may take a while for large files...")
    
    try:
        # Load directly from CSV using datasets library
        print("   Loading CSV file directly...")
        from datasets import load_dataset
        
        # Load dataset from CSV
        dataset = load_dataset("csv", data_files=str(DATASET_PATH), split="train")
        
        print(f"   ✅ Loaded {len(dataset):,} rows")
        print(f"   Columns: {', '.join(dataset.column_names)}")
        
        # Create DatasetDict with train split
        dataset_dict = DatasetDict({"train": dataset})
        
        print("✅ Dataset converted successfully!")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\n💡 Trying alternative method with pandas...")
        try:
            # Fallback: use pandas and convert
            print("   Reading CSV file with pandas...")
            df = pd.read_csv(DATASET_PATH, nrows=1000)  # Sample first
            print(f"   Sample columns: {', '.join(df.columns.tolist())}")
            print("   Loading full dataset...")
            df = pd.read_csv(DATASET_PATH)
            print(f"   ✅ Loaded {len(df):,} rows")
            
            # Convert to Hugging Face Dataset
            print("\n🔄 Converting to Hugging Face Dataset format...")
            dataset = Dataset.from_pandas(df)
            dataset_dict = DatasetDict({"train": dataset})
            print("✅ Dataset converted successfully!")
        except Exception as e2:
            print(f"❌ Alternative method also failed: {e2}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Upload to Hugging Face
    print(f"\n📤 Uploading to Hugging Face Hub...")
    print(f"   Repository: {REPO_ID}")
    print("   This may take several minutes for large datasets...")
    
    # Try uploading as a proper dataset first, but fall back to CSV if there are pyarrow issues
    upload_success = False
    
    try:
        # Try with max_shard_size to avoid pyarrow compatibility issues
        # Disable content-defined chunking which requires newer pyarrow
        os.environ['HF_DATASETS_USE_CONTENT_DEFINED_CHUNKING'] = 'false'
        
        dataset_dict.push_to_hub(
            repo_id=REPO_ID,
            private=False,  # Set to True if you want a private dataset
            max_shard_size="500MB",  # Limit shard size to avoid issues
        )
        print("\n✅ Dataset uploaded successfully!")
        upload_success = True
        
    except Exception as e:
        error_msg = str(e)
        if 'use_content_defined_chunking' in error_msg or 'pyarrow' in error_msg.lower():
            print(f"\n⚠️  PyArrow compatibility issue detected: {error_msg}")
            print("   Falling back to direct CSV upload method...")
        else:
            print(f"\n⚠️  Error uploading dataset: {e}")
            print("   Trying alternative upload method...")
        
        # Alternative: Upload CSV file directly using Hugging Face Hub API
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            print("   Uploading CSV file directly...")
            api.upload_file(
                path_or_fileobj=str(DATASET_PATH),
                path_in_repo="data.csv",
                repo_id=REPO_ID,
                repo_type="dataset",
            )
            print("\n✅ Dataset uploaded successfully (as CSV file)!")
            print("   Note: The dataset is uploaded as a CSV file.")
            print("   Users can load it with:")
            print(f"   from datasets import load_dataset")
            print(f"   dataset = load_dataset('csv', data_files='{REPO_ID}/data.csv', repo_id='{REPO_ID}')")
            upload_success = True
        except Exception as e2:
            print(f"\n❌ Alternative method also failed: {e2}")
            print("\n💡 Troubleshooting:")
            print("   - Make sure you have write access to the repository")
            print("   - Check your internet connection")
            print("   - The repository may need to be created first")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    if not upload_success:
        print("\n❌ Upload failed with all methods")
        sys.exit(1)
    
    # Upload README.md (dataset card)
    print(f"\n📝 Uploading dataset card (README.md)...")
    readme_path = Path(__file__).parent.parent / "data" / "combined" / "README.md"
    if readme_path.exists():
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=REPO_ID,
                repo_type="dataset",
            )
            print("✅ Dataset card uploaded successfully!")
        except Exception as e:
            print(f"⚠️  Warning: Could not upload README.md: {e}")
    else:
        print("⚠️  Warning: README.md not found. Dataset card will not be uploaded.")
        print(f"   Expected location: {readme_path}")
    
    # Display summary
    print("\n" + "=" * 60)
    print("✅ UPLOAD COMPLETE!")
    print("=" * 60)
    print(f"\n🌐 View your dataset at:")
    print(f"   https://huggingface.co/datasets/{REPO_ID}")
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total rows: {len(dataset_dict['train']):,}")
    print(f"   Columns: {len(dataset_dict['train'].column_names)}")
    print(f"   Size: {DATASET_PATH.stat().st_size / (1024*1024):.1f} MB")
    
    print(f"\n💡 To use this dataset:")
    print(f"   from datasets import load_dataset")
    print(f"   dataset = load_dataset('{REPO_ID}')")
    
    print(f"\n✨ Your dataset is now available on Hugging Face!")

if __name__ == "__main__":
    main()

