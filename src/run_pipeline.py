#!/usr/bin/env python3
"""
Religious Text Scraper & NLP Pipeline
======================================
Main entry point for the resumable async crawler with live dashboard.

Output files by religion:
- Scraped_Catholic.csv
- Scraped_Islam.csv
- Scraped_Protestant.csv
- Rejected_Non_Indonesian.csv
"""

import argparse
import asyncio
import csv
import io
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List

import aiofiles
from rich.console import Console

from config import (
    BASE_DIR,
    SAVE_INTERVAL,
    FILE_ENCODING,
    HISTORY_LOG,
    CONTENT_HASHES_FILE,
    QUEUE_FILE,
    DEPTH_BOUNDARY_FILE,
    STATS_FILE,
)
from crawler import AsyncCrawler
from dashboard import LiveDashboard, print_final_summary


console = Console()

# CSV field names for output files
CSV_FIELDNAMES = ['Label', 'Denomination', 'Location', 'Date', 'Title', 'Sentence_Unit', 'Link']

# Output file paths
OUTPUT_FILES = {
    'Catholic': BASE_DIR / 'data' / 'scraped' / 'Scraped_Catholic.csv',
    'Islam': BASE_DIR / 'data' / 'scraped' / 'Scraped_Islam.csv',
    'Protestant': BASE_DIR / 'data' / 'scraped' / 'Scraped_Protestant.csv',
}
REJECTED_FILE = BASE_DIR / 'data' / 'scraped' / 'Rejected_Non_Indonesian.csv'

# Lock for CSV writing to prevent race conditions
_csv_write_lock = asyncio.Lock()


def map_to_output_format(sentence_data: dict) -> dict:
    """Map internal format to output CSV format."""
    return {
        'Label': sentence_data.get('religion', ''),
        'Denomination': sentence_data.get('source', ''),
        'Location': sentence_data.get('region', ''),
        'Date': '',  # Not available from scraping
        'Title': '',  # Could be extracted but not implemented
        'Sentence_Unit': sentence_data.get('sentence', ''),
        'Link': sentence_data.get('url', ''),
    }


async def save_to_csv_file(file_path: Path, sentences: List[dict], fieldnames: List[str]):
    """Save sentences to a specific CSV file."""
    if not sentences:
        return
    
    file_exists = file_path.exists()
    
    # Build CSV content in memory
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
    
    if not file_exists:
        writer.writeheader()
    
    # Map to output format
    mapped_sentences = [map_to_output_format(s) for s in sentences]
    writer.writerows(mapped_sentences)
    csv_content = output.getvalue()
    
    # Write asynchronously
    try:
        async with aiofiles.open(file_path, mode='a', encoding=FILE_ENCODING) as f:
            await f.write(csv_content)
    except Exception as e:
        console.print(f"[red]⚠️  Error saving {file_path.name}: {e}[/red]")


async def save_sentences_by_religion(
    sentences_by_religion: Dict[str, List[dict]], 
    rejected: List[dict]
):
    """Save sentences to separate files by religion."""
    async with _csv_write_lock:
        # Save by religion
        for religion, sentences in sentences_by_religion.items():
            if sentences:
                file_path = OUTPUT_FILES.get(religion)
                if file_path:
                    await save_to_csv_file(file_path, sentences, CSV_FIELDNAMES)
                else:
                    # Unknown religion, save to a generic file
                    generic_path = BASE_DIR / 'data' / 'scraped' / f'Scraped_{religion}.csv'
                    generic_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                    await save_to_csv_file(generic_path, sentences, CSV_FIELDNAMES)
        
        # Save rejected sentences
        if rejected:
            # Add detected language for debugging
            rejected_fieldnames = CSV_FIELDNAMES + ['Detected_Language']
            
            file_exists = REJECTED_FILE.exists()
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=rejected_fieldnames, extrasaction='ignore')
            
            if not file_exists:
                writer.writeheader()
            
            for s in rejected:
                row = map_to_output_format(s)
                row['Detected_Language'] = s.get('detected_lang', 'unknown')
                writer.writerow(row)
            
            try:
                async with aiofiles.open(REJECTED_FILE, mode='a', encoding=FILE_ENCODING) as f:
                    await f.write(output.getvalue())
            except Exception as e:
                console.print(f"[red]⚠️  Error saving rejected: {e}[/red]")


async def save_sentences_periodically(crawler: AsyncCrawler, interval: int = SAVE_INTERVAL):
    """Periodically save collected sentences to CSVs and queue state."""
    last_count = 0
    last_queue_save = 0
    
    while True:
        await asyncio.sleep(5)
        
        current_count = crawler.stats.total_sites
        if current_count - last_count >= interval:
            by_religion, rejected = await crawler.get_sentences_batch()
            if any(by_religion.values()) or rejected:
                await save_sentences_by_religion(by_religion, rejected)
            
            last_count = current_count
        
        # Save queue every 10 seconds (more frequent than CSV saves)
        # This ensures queue persists even if interrupted
        current_time = time.time()
        if current_time - last_queue_save >= 10:
            await crawler._save_queue()
            last_queue_save = current_time


async def run_with_dashboard(crawler: AsyncCrawler, num_workers: int):
    """Run crawler with live dashboard."""
    dashboard = LiveDashboard(crawler)
    
    crawler_task = asyncio.create_task(crawler.run(num_workers))
    dashboard_task = asyncio.create_task(dashboard.run())
    saver_task = asyncio.create_task(save_sentences_periodically(crawler))
    
    def shutdown_handler():
        """Handle shutdown signal - stop crawler."""
        console.print("\n[yellow]⚠️  Shutting down...[/yellow]")
        crawler.stop()
        dashboard.stop()
        saver_task.cancel()
    
    if sys.platform != 'win32':
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_handler)
    
    try:
        await crawler_task
    except asyncio.CancelledError:
        pass
    finally:
        dashboard.stop()
        saver_task.cancel()
        
        try:
            await asyncio.wait_for(dashboard_task, timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        
        # Final save - ensure queue is saved
        await crawler._save_queue()
        by_religion, rejected = await crawler.get_sentences_batch()
        await save_sentences_by_religion(by_religion, rejected)


async def run_without_dashboard(crawler: AsyncCrawler, num_workers: int):
    """Run crawler with simple progress output."""
    from tqdm.asyncio import tqdm
    
    saver_task = asyncio.create_task(save_sentences_periodically(crawler))
    
    console.print("[bold cyan]🚀 Starting crawler...[/bold cyan]")
    console.print(f"   Workers: {num_workers}")
    console.print(f"   Queue: {crawler.stats.queue_size} URLs")
    console.print(f"   Already visited: {len(crawler.visited)} URLs\n")
    
    async def progress_display():
        with tqdm(desc="Crawling", unit=" URLs") as pbar:
            last_count = 0
            while crawler._running or not crawler.queue.empty():
                current = crawler.stats.total_sites
                pbar.update(current - last_count)
                pbar.set_postfix({
                    'sentences': crawler.stats.total_sentences,
                    'queue': crawler.stats.queue_size
                })
                last_count = current
                await asyncio.sleep(0.5)
    
    progress_task = asyncio.create_task(progress_display())
    
    try:
        await crawler.run(num_workers)
    except asyncio.CancelledError:
        pass
    finally:
        saver_task.cancel()
        progress_task.cancel()
        
        # Final save - ensure queue is saved
        await crawler._save_queue()
        by_religion, rejected = await crawler.get_sentences_batch()
        await save_sentences_by_religion(by_religion, rejected)


def delete_file_safely(file_path: Path) -> bool:
    """Safely delete a file if it exists."""
    path = Path(file_path)
    if path.exists():
        path.unlink()
        return True
    return False


def archive_files(file_paths: List[Path], archive_name: str):
    """Move files to an archive folder with timestamp."""
    from datetime import datetime
    
    # Create archive directory
    archive_dir = BASE_DIR / "archive" / archive_name
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    archived = []
    for file_path in file_paths:
        if file_path.exists():
            dest = archive_dir / file_path.name
            # If file already exists in archive, add timestamp
            if dest.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                stem = file_path.stem
                suffix = file_path.suffix
                dest = archive_dir / f"{stem}_{timestamp}{suffix}"
            
            file_path.rename(dest)
            archived.append(file_path.name)
    
    return archive_dir, archived


async def main(args):
    """Main entry point."""
    console.print("""
[bold blue]╔══════════════════════════════════════════════════════════════╗
║     [cyan]Religious Text Scraper & NLP Pipeline[/cyan]                    ║
║     [dim]Master Specification Implementation[/dim]                       ║
╚══════════════════════════════════════════════════════════════╝[/bold blue]
    """)
    
    # Handle --fresh flag
    if args.fresh:
        console.print("[yellow]🗑️  --fresh flag: Archiving old data and clearing...[/yellow]")
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"fresh_{timestamp}"
        
        files_to_archive = [
            HISTORY_LOG, 
            CONTENT_HASHES_FILE,
            QUEUE_FILE,
            DEPTH_BOUNDARY_FILE,
            STATS_FILE,
            *OUTPUT_FILES.values(),
            REJECTED_FILE,
        ]
        
        archive_dir, archived = archive_files(files_to_archive, archive_name)
        if archived:
            console.print(f"   📦 Archived {len(archived)} files to: archive/{archive_name}/")
            for fname in archived[:5]:  # Show first 5
                console.print(f"      • {fname}")
            if len(archived) > 5:
                console.print(f"      ... and {len(archived) - 5} more")
        console.print()
    
    # Handle --extend flag
    elif args.extend:
        console.print("[yellow]🔄 --extend flag: Archiving history to revisit sites...[/yellow]")
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"extend_{timestamp}"
        
        files_to_archive = [HISTORY_LOG, CONTENT_HASHES_FILE, QUEUE_FILE]
        
        archive_dir, archived = archive_files(files_to_archive, archive_name)
        if archived:
            console.print(f"   📦 Archived {len(archived)} files to: archive/{archive_name}/")
            for fname in archived:
                console.print(f"      • {fname}")
        console.print()
    
    # Handle --depth flag
    if args.depth:
        import config
        old_depth = config.MAX_DEPTH
        config.MAX_DEPTH = args.depth
        console.print(f"[cyan]📏 Depth override: {old_depth} → {args.depth}[/cyan]\n")
    
    # Initialize crawler
    crawler = AsyncCrawler()
    await crawler.initialize()
    
    console.print(f"📋 Loaded [green]{crawler.queue.qsize()}[/green] seed URLs")
    console.print(f"📂 History: [green]{len(crawler.visited)}[/green] URLs already visited")
    if DEPTH_BOUNDARY_FILE.exists():
        try:
            import json
            with open(DEPTH_BOUNDARY_FILE, 'r', encoding=FILE_ENCODING) as f:
                boundary_data = json.load(f)
                console.print(f"🔗 Depth boundary URLs: [green]{len(boundary_data)}[/green] URLs at MAX_DEPTH (for continuation)")
        except Exception:
            pass
    console.print()
    console.print("[dim]Output files:[/dim]")
    for religion, path in OUTPUT_FILES.items():
        console.print(f"   {religion}: {path.name}")
    console.print(f"   Rejected: {REJECTED_FILE.name}")
    console.print(f"   Depth boundary: {DEPTH_BOUNDARY_FILE.name} (for continuation)")
    console.print()
    
    try:
        if args.no_dashboard:
            await run_without_dashboard(crawler, args.workers)
        else:
            await run_with_dashboard(crawler, args.workers)
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Interrupted by user[/yellow]")
    finally:
        await crawler.close()
        print_final_summary(crawler, console)
        console.print("\n[green]✅ Progress saved. Run again to resume.[/green]")


def cli():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Religious Text Scraper & NLP Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Normal run (resumes from history)
  python run_pipeline.py --fresh            # Start fresh, clear all history
  python run_pipeline.py --extend           # Go deeper on existing sites
  python run_pipeline.py --depth 12         # Set custom depth
        """
    )
    
    parser.add_argument('--no-dashboard', action='store_true',
                        help='Run without the live dashboard')
    parser.add_argument('--workers', type=int, default=267,
                        help='Number of parallel workers (default: 267, reduced by 1/3 - slows down on server errors)')
    parser.add_argument('--depth', type=int, default=None,
                        help='Override MAX_DEPTH (default: 8)')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh: clear all data files')
    parser.add_argument('--extend', action='store_true',
                        help='Extend crawl: keep CSV data but clear history')
    
    args = parser.parse_args()
    
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
