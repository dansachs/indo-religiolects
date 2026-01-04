#!/usr/bin/env python3
"""
Upload the trained model_final to Hugging Face.

Usage:
    python scripts/upload_model_to_hf.py
"""

import os
import sys
from pathlib import Path
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError

# Configuration
BASE_DIR = Path(__file__).parent.parent.resolve()
MODEL_DIR = BASE_DIR / "models" / "trained" / "model_final"
REPO_ID = "dansachs/indo-religiolect-bert-v2"

def main():
    print("=" * 60)
    print("📤 Uploading Model to Hugging Face")
    print("=" * 60)
    
    # Check if model directory exists
    if not MODEL_DIR.exists():
        print(f"❌ Model directory not found: {MODEL_DIR}")
        sys.exit(1)
    
    # Check for required files
    required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not (MODEL_DIR / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        sys.exit(1)
    
    print(f"\n📂 Model directory: {MODEL_DIR}")
    print(f"📦 Repository: {REPO_ID}")
    
    # List files to upload
    files_to_upload = list(MODEL_DIR.glob("*"))
    files_to_upload = [f for f in files_to_upload if f.is_file()]
    print(f"\n📄 Files to upload ({len(files_to_upload)}):")
    for file in files_to_upload:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   - {file.name} ({size_mb:.2f} MB)")
    
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
    
    # Initialize Hugging Face API
    api = HfApi()
    
    # Create repository if it doesn't exist
    print(f"\n🔍 Checking repository: {REPO_ID}")
    try:
        api.repo_info(repo_id=REPO_ID, repo_type="model")
        print("✅ Repository exists")
        print("   ℹ️  Note: If you want a fresh upload, delete the repository first")
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            print("📦 Repository not found. Creating new V2 repository...")
            try:
                create_repo(
                    repo_id=REPO_ID,
                    repo_type="model",
                    private=False,
                    exist_ok=False
                )
                print("✅ Repository created successfully!")
            except Exception as e2:
                print(f"❌ Failed to create repository: {e2}")
                sys.exit(1)
        else:
            print(f"❌ Error checking repository: {e}")
            sys.exit(1)
    
    # Filter out image files (we'll upload the rest to repo root)
    # Include README.md if it exists in model_final, otherwise we'll create one
    files_to_upload_filtered = [
        f for f in files_to_upload 
        if f.suffix.lower() not in ['.png', '.jpg', '.jpeg']
    ]
    
    # Ensure README.md exists (it should be in model_final now)
    readme_path = MODEL_DIR / "README.md"
    if not readme_path.exists():
        print("⚠️  Warning: README.md not found in model_final directory")
        print("   The model card will not be uploaded.")
    else:
        if readme_path not in files_to_upload_filtered:
            files_to_upload_filtered.append(readme_path)
        print(f"   ✅ README.md found and will be uploaded")
    
    # Upload files
    print(f"\n📤 Uploading model files to Hugging Face Hub...")
    print(f"   Repository: {REPO_ID}")
    print("   This may take several minutes for large files...")
    
    try:
        # Upload each file individually to the root of the repository
        for i, file_path in enumerate(files_to_upload_filtered, 1):
            print(f"   [{i}/{len(files_to_upload_filtered)}] Uploading {file_path.name}...")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_path.name,  # Upload to root, not in a subdirectory
                repo_id=REPO_ID,
                repo_type="model",
            )
        print("\n✅ Model uploaded successfully!")
        
    except Exception as e:
        print(f"\n❌ Error uploading model: {e}")
        print("\n💡 Troubleshooting:")
        print("   - Make sure you have write access to the repository")
        print("   - Check your internet connection")
        print("   - Verify the repository exists and is accessible")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Display summary
    print("\n" + "=" * 60)
    print("✅ UPLOAD COMPLETE!")
    print("=" * 60)
    print(f"\n🌐 View your model at:")
    print(f"   https://huggingface.co/{REPO_ID}")
    print(f"\n📊 Uploaded Files:")
    for file in files_to_upload_filtered:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   - {file.name} ({size_mb:.2f} MB)")
    
    print(f"\n💡 To use this model:")
    print(f"   from transformers import AutoTokenizer, AutoModelForSequenceClassification")
    print(f"   tokenizer = AutoTokenizer.from_pretrained('{REPO_ID}')")
    print(f"   model = AutoModelForSequenceClassification.from_pretrained('{REPO_ID}')")
    
    print(f"\n✨ Your model is now available on Hugging Face!")

if __name__ == "__main__":
    main()

