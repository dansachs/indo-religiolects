#!/usr/bin/env python3
"""
Helper script to add depth boundary URLs to seeds.json for continuation.

This allows you to continue crawling deeper by using URLs that were
discovered at MAX_DEPTH in a previous run.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.resolve()  # Go up one level from scripts/ to project root
SEEDS_FILE = BASE_DIR / "configs" / "seeds.json"
DEPTH_BOUNDARY_FILE = BASE_DIR / "data" / "crawler_state" / "depth_boundary_urls.json"
HISTORY_LOG = BASE_DIR / "data" / "crawler_state" / "history.log"


def add_boundary_urls_to_seeds():
    """Add boundary URLs to seeds.json for continuation."""
    # Load boundary URLs
    if not DEPTH_BOUNDARY_FILE.exists():
        print(f"❌ {DEPTH_BOUNDARY_FILE.name} not found!")
        print("   Run the crawler first to generate boundary URLs at MAX_DEPTH.")
        sys.exit(1)
    
    with open(DEPTH_BOUNDARY_FILE, 'r', encoding='utf-8') as f:
        boundary_urls = json.load(f)
    
    if not boundary_urls:
        print(f"⚠️  {DEPTH_BOUNDARY_FILE.name} is empty.")
        print("   No boundary URLs found. The crawler may not have reached MAX_DEPTH yet.")
        sys.exit(1)
    
    print(f"📋 Found {len(boundary_urls)} boundary URLs")
    
    # Load visited URLs from history.log (to skip already-scraped URLs)
    visited_urls = set()
    if HISTORY_LOG.exists():
        with open(HISTORY_LOG, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                if url:
                    visited_urls.add(url)
        print(f"📂 Found {len(visited_urls)} already-visited URLs in history.log")
    
    # Load existing seeds
    with open(SEEDS_FILE, 'r', encoding='utf-8') as f:
        seeds_data = json.load(f)
    
    # Group boundary URLs by religion
    added_count = 0
    skipped_visited = 0
    for entry in boundary_urls:
        url = entry['url']
        religion = entry['religion'].lower()
        
        # Skip if already visited
        if url in visited_urls:
            skipped_visited += 1
            continue
        
        # Check if already in seeds
        already_exists = False
        if religion in seeds_data:
            for site in seeds_data[religion]:
                if site['url'] == url:
                    already_exists = True
                    break
        
        if not already_exists:
            if religion not in seeds_data:
                seeds_data[religion] = []
            
            seeds_data[religion].append({
                'url': url,
                'source': entry.get('source', 'Depth_Boundary'),
                'region': entry.get('region', 'Unknown'),
            })
            added_count += 1
    
    # Save updated seeds
    with open(SEEDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(seeds_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Added {added_count} new URLs to seeds.json")
    skipped_total = len(boundary_urls) - added_count
    if skipped_visited > 0:
        print(f"   (Skipped {skipped_visited} already-visited, {skipped_total - skipped_visited} duplicates)")
    else:
        print(f"   (Skipped {skipped_total} duplicates)")
    print()
    print("📌 Note: The crawler will automatically skip URLs that are already in history.log")
    print("   So even if added to seeds.json, visited URLs won't be re-scraped.")
    print()
    print("Now you can run the crawler with increased depth:")
    print("   python run_pipeline.py --depth 10  # or higher")


if __name__ == "__main__":
    add_boundary_urls_to_seeds()

