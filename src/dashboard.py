"""
Live dashboard for monitoring crawling progress using Rich.
Includes adaptive rate limiting status display.
"""

import asyncio
from datetime import timedelta
from typing import TYPE_CHECKING

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from crawler import AsyncCrawler


class LiveDashboard:
    """
    Real-time terminal dashboard for monitoring crawler progress.
    Includes adaptive rate limiting status.
    """
    
    def __init__(self, crawler: "AsyncCrawler"):
        self.crawler = crawler
        self.console = Console()
        self._running = False
    
    def _format_duration(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        td = timedelta(seconds=int(seconds))
        return str(td)
    
    def _make_header(self) -> Panel:
        """Create the header panel."""
        header_text = Text()
        header_text.append("🕌 ", style="bold")
        header_text.append("Religious Text Scraper & NLP Pipeline", style="bold cyan")
        header_text.append(" ⛪", style="bold")
        return Panel(header_text, style="bold white on dark_blue")
    
    def _make_stats_table(self) -> Table:
        """Create the main statistics table."""
        stats = self.crawler.stats
        
        table = Table(title="📊 Scraping Statistics", expand=True, 
                      title_style="bold magenta")
        
        table.add_column("Religion", style="cyan", justify="left")
        table.add_column("Sites Scraped", style="green", justify="right")
        table.add_column("Sentences", style="yellow", justify="right")
        table.add_column("Progress", justify="left")
        
        religions = ['Catholic', 'Islam', 'Protestant']
        max_sentences = max(stats.sentences_collected.values()) if stats.sentences_collected else 1
        
        for religion in religions:
            sites = stats.sites_scraped.get(religion, 0)
            sentences = stats.sentences_collected.get(religion, 0)
            
            # Simple progress bar
            bar_width = 20
            filled = int((sentences / max(max_sentences, 1)) * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            
            table.add_row(
                religion,
                f"{sites:,}",
                f"{sentences:,}",
                f"[green]{bar}[/green]"
            )
        
        # Total row
        table.add_section()
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{stats.total_sites:,}[/bold]",
            f"[bold]{stats.total_sentences:,}[/bold]",
            ""
        )
        
        return table
    
    def _make_status_panel(self) -> Panel:
        """Create the status panel with time estimates."""
        stats = self.crawler.stats
        
        status_table = Table.grid(padding=(0, 2))
        status_table.add_column(justify="right", style="bold")
        status_table.add_column(justify="left")
        
        # Progress bar
        progress = stats.progress_percent
        bar_width = 15
        filled = int((progress / 100) * bar_width)
        progress_bar = "█" * filled + "░" * (bar_width - filled)
        progress_text = Text()
        progress_text.append(f"{progress_bar} ", style="green")
        progress_text.append(f"{progress:.1f}%", style="bold green")
        status_table.add_row("📊 Progress:", progress_text)
        
        # Time stats - show both current session and total runtime
        status_table.add_row("⏱️  Session:", self._format_duration(stats.elapsed_time))
        status_table.add_row("⏱️  Total:", self._format_duration(stats.total_elapsed_time))
        
        # ETA
        remaining = stats.estimated_remaining_seconds
        if remaining > 0:
            eta_text = Text(self._format_duration(remaining), style="cyan")
        else:
            eta_text = Text("Calculating...", style="dim")
        status_table.add_row("⏳ ETA:", eta_text)
        
        # Speed
        status_table.add_row("⚡ Speed:", f"{stats.urls_per_minute:.1f} URLs/min")
        
        # Queue and workers
        status_table.add_row("📝 Queue:", f"{stats.queue_size:,} URLs")
        status_table.add_row("👷 Workers:", f"{stats.active_workers} active")
        status_table.add_row("🔄 Domains:", f"{self.crawler.queue.domain_count()} active")
        
        return Panel(status_table, title="🔧 Status", border_style="blue")
    
    def _make_rate_limit_panel(self) -> Panel:
        """Create the adaptive rate limiting status panel."""
        stats = self.crawler.stats
        
        rate_table = Table.grid(padding=(0, 2))
        rate_table.add_column(justify="right", style="bold")
        rate_table.add_column(justify="left")
        
        # Cooling down domains
        cooling_count = len(stats.domains_cooling_down)
        if cooling_count > 0:
            cooling_text = Text(f"{cooling_count} ", style="bold red")
            cooling_text.append("domains", style="red")
            rate_table.add_row("🧊 Cooling:", cooling_text)
            
            # Show first few cooling domains
            for domain in stats.domains_cooling_down[:2]:
                domain_short = domain[:20] + "..." if len(domain) > 20 else domain
                rate_table.add_row("", Text(f"  ↳ {domain_short}", style="dim red"))
        else:
            rate_table.add_row("🧊 Cooling:", Text("None", style="green"))
        
        # Slowed domains
        slowed_count = len(stats.domains_slowed)
        if slowed_count > 0:
            slowed_text = Text(f"{slowed_count} ", style="bold yellow")
            slowed_text.append("domains", style="yellow")
            rate_table.add_row("🐢 Slowed:", slowed_text)
        else:
            rate_table.add_row("🐢 Slowed:", Text("None", style="green"))
        
        # Counts summary
        rate_table.add_row("❌ Failed:", f"{stats.failed_urls:,}")
        rate_table.add_row("🔁 Deduped:", f"{stats.skipped_duplicates:,}")
        rate_table.add_row("⚠️  Errors:", f"{stats.total_429_errors + stats.total_503_errors:,}")
        
        return Panel(rate_table, title="🚦 Health", border_style="yellow")
    
    def _make_activity_panel(self) -> Panel:
        """Create the current activity panel."""
        stats = self.crawler.stats
        
        if stats.current_urls:
            activity_text = Text()
            for i, url in enumerate(stats.current_urls[-5:]):
                if i > 0:
                    activity_text.append("\n")
                activity_text.append("→ ", style="green")
                activity_text.append(url, style="dim")
        else:
            activity_text = Text("Waiting for tasks...", style="dim italic")
        
        return Panel(activity_text, title="🌐 Current Activity", border_style="green")
    
    def _make_config_panel(self) -> Panel:
        """Create the configuration panel."""
        from config import MAX_DEPTH, MAX_PAGES_PER_DOMAIN, NLP_WORKERS
        from crawler import MAX_GLOBAL_CONCURRENT, MAX_PER_DOMAIN_CONCURRENT
        import os
        
        nlp_workers = NLP_WORKERS or max(1, (os.cpu_count() or 4) - 1)
        
        config_table = Table.grid(padding=(0, 2))
        config_table.add_column(justify="right", style="dim")
        config_table.add_column(justify="left")
        
        config_table.add_row("Max Depth:", f"{MAX_DEPTH}")
        config_table.add_row("Global Limit:", f"{MAX_GLOBAL_CONCURRENT}")
        config_table.add_row("Per Domain:", f"{MAX_PER_DOMAIN_CONCURRENT}")
        config_table.add_row("NLP Workers:", f"{nlp_workers}")
        
        return Panel(config_table, title="⚙️  Config", border_style="dim")
    
    def generate_layout(self) -> Layout:
        """Generate the full dashboard layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=8)
        )
        
        layout["body"].split_row(
            Layout(name="main", ratio=2),
            Layout(name="sidebar", ratio=1)
        )
        
        layout["sidebar"].split_column(
            Layout(name="status"),
            Layout(name="rate_limit"),
            Layout(name="config", size=6)
        )
        
        # Populate layout
        layout["header"].update(self._make_header())
        layout["main"].update(self._make_stats_table())
        layout["status"].update(self._make_status_panel())
        layout["rate_limit"].update(self._make_rate_limit_panel())
        layout["config"].update(self._make_config_panel())
        layout["footer"].update(self._make_activity_panel())
        
        return layout
    
    async def run(self, refresh_rate: float = 0.5):
        """Run the live dashboard."""
        self._running = True
        
        with Live(self.generate_layout(), console=self.console, 
                  refresh_per_second=int(1/refresh_rate), screen=True) as live:
            while self._running:
                live.update(self.generate_layout())
                await asyncio.sleep(refresh_rate)
    
    def stop(self):
        """Stop the dashboard."""
        self._running = False


def print_final_summary(crawler: "AsyncCrawler", console: Console = None):
    """Print final summary after crawling completes."""
    if console is None:
        console = Console()
    
    stats = crawler.stats
    
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]✅ Crawling Complete![/bold green]",
        border_style="green"
    ))
    
    # Final stats table
    table = Table(title="📊 Final Results", expand=False)
    table.add_column("Religion", style="cyan")
    table.add_column("Sites", style="green", justify="right")
    table.add_column("Sentences", style="yellow", justify="right")
    
    for religion in ['Catholic', 'Islam', 'Protestant']:
        table.add_row(
            religion,
            f"{stats.sites_scraped.get(religion, 0):,}",
            f"{stats.sentences_collected.get(religion, 0):,}"
        )
    
    table.add_section()
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{stats.total_sites:,}[/bold]",
        f"[bold]{stats.total_sentences:,}[/bold]"
    )
    
    console.print(table)
    
    # Rate limiting summary
    if stats.total_429_errors > 0 or stats.total_503_errors > 0:
        console.print("\n[yellow]🚦 Rate Limiting Summary:[/yellow]")
        console.print(f"   429 Errors: {stats.total_429_errors:,}")
        console.print(f"   503 Errors: {stats.total_503_errors:,}")
        console.print(f"   Total Retries: {stats.total_retries:,}")
        console.print(f"   Domains Slowed: {len(stats.domains_slowed)}")
    
    # Summary info
    console.print(f"\n⏱️  Session runtime: {timedelta(seconds=int(stats.elapsed_time))}")
    console.print(f"⏱️  Total runtime (all runs): {timedelta(seconds=int(stats.total_elapsed_time))}")
    console.print(f"⚡ Average speed: {stats.urls_per_minute:.1f} URLs/minute")
    console.print(f"❌ Failed URLs: {stats.failed_urls:,}")
    console.print(f"\n📁 Output: [cyan]religious_corpus.csv[/cyan]")
    console.print(f"📂 History: [cyan]history.log[/cyan]")
