#!/usr/bin/env python3
"""
Main entry point for the Religious Text Scraper.

This script runs the crawler from the project root.
Usage: python run_crawler.py [options]
"""

import sys
from pathlib import Path

# Add src directory to path so imports work
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Import and run the main function
from run_pipeline import main, cli

if __name__ == "__main__":
    args = cli()
    import asyncio
    asyncio.run(main(args))

