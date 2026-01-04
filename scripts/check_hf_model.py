#!/usr/bin/env python3
"""Check what files are in the Hugging Face repository."""

from huggingface_hub import HfApi

REPO_ID = "dansachs/indo-religiolect-bert"

api = HfApi()

print("=" * 60)
print(f"Checking repository: {REPO_ID}")
print("=" * 60)

try:
    # Get repository info
    info = api.repo_info(repo_id=REPO_ID, repo_type="model")
    print(f"\n✅ Repository exists")
    print(f"   Private: {info.private}")
    print(f"   Last modified: {info.last_modified}")
    print(f"   Total files: {len(info.siblings)}")
    
    # List all files
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="model")
    print(f"\n📄 Files in repository ({len(files)}):")
    for f in sorted(files):
        print(f"   ✓ {f}")
    
    # Check for key files
    required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "label_map.json"
    ]
    
    print(f"\n🔍 Checking required files:")
    missing = []
    for req_file in required_files:
        if req_file in files:
            print(f"   ✅ {req_file}")
        else:
            print(f"   ❌ {req_file} (MISSING)")
            missing.append(req_file)
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
    else:
        print(f"\n✅ All required files are present!")
    
    print(f"\n🌐 View repository at:")
    print(f"   https://huggingface.co/{REPO_ID}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

