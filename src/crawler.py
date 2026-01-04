"""
Async web crawler with ethical, adaptive per-domain rate limiting.
Uses ProcessPoolExecutor for CPU-bound NLP to avoid blocking the event loop.

Features:
- Per-domain throttling (1 concurrent request per domain by default)
- Global limit (50 total concurrent requests to save RAM)
- Adaptive backoff on 429/503 errors (60s pause + permanent slowdown)
- Domain rotation for maximum parallelism
- User-agent rotation
- URL trap detection
- robots.txt respect
"""

import asyncio
import hashlib
import io
import json
import os
import random
import re
import ssl
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import aiohttp
import aiofiles

from config import (
    SEEDS_FILE,
    HISTORY_LOG,
    CONTENT_HASHES_FILE,
    QUEUE_FILE,
    DEPTH_BOUNDARY_FILE,
    STATS_FILE,
    MAX_DEPTH,
    MAX_CONCURRENT_REQUESTS,
    REQUEST_TIMEOUT,
    DELAY_BETWEEN_REQUESTS,
    MIN_DELAY,
    MAX_DELAY,
    MAX_PAGES_PER_DOMAIN,
    REQUEST_HEADERS,
    CONNECTION_LIMIT,
    CONNECTION_LIMIT_PER_HOST,
    DNS_CACHE_TTL,
    KEEPALIVE_TIMEOUT,
    ENABLE_DEDUPLICATION,
    MIN_CONTENT_LENGTH,
    NLP_WORKERS,
    FILE_ENCODING,
    BASE_DIR,
)

# Note: extract_sentences is imported inside _extract_sentences_async()
# to avoid multiprocessing pickle issues

# =============================================================================
# USER-AGENT ROTATION
# =============================================================================

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

def get_random_user_agent() -> str:
    """Get a random user agent for request rotation."""
    return random.choice(USER_AGENTS)


# =============================================================================
# URL TRAP DETECTION
# =============================================================================

SKIP_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.doc', 
                   '.docx', '.xls', '.xlsx', '.mp3', '.mp4', '.avi', '.mov',
                   '.css', '.js', '.ico', '.svg', '.woff', '.woff2', '.ttf', '.eot'}

def is_url_trap(url: str) -> bool:
    """
    Detect URL traps that could cause infinite crawling.
    
    Traps detected:
    - URLs with >6 path segments
    - URLs with repeating segments (e.g., /news/news/news)
    - URLs with skip extensions
    """
    parsed = urlparse(url)
    path = parsed.path
    
    # Check extension
    lower_path = path.lower()
    for ext in SKIP_EXTENSIONS:
        if lower_path.endswith(ext):
            return True
    
    # Check path segments
    segments = [s for s in path.split('/') if s]
    
    # Too many segments
    if len(segments) > 6:
        return True
    
    # Repeating segments
    if len(segments) >= 2:
        for i in range(len(segments) - 1):
            if segments[i] == segments[i + 1]:
                return True
    
    # Check for calendar/pagination traps
    if re.search(r'/\d{4}/\d{2}/\d{2}/\d{4}/', path):
        return True
    
    # Skip non-content paths (feeds, APIs, etc.)
    skip_paths = ['/feed', '/rss', '/sitemap', '/api/', '/ajax/', '/wp-json/', 
                  '/wp-admin/', '/wp-includes/', '/cgi-bin/', '/.well-known/']
    if any(skip in lower_path for skip in skip_paths):
        return True
    
    return False


# =============================================================================
# CONTENT HASHING
# =============================================================================

def hash_content(content: str) -> str:
    """Generate MD5 hash of cleaned content for deduplication."""
    return hashlib.md5(content.encode('utf-8', errors='ignore')).hexdigest()


# =============================================================================
# ADAPTIVE RATE LIMITING CONFIGURATION
# =============================================================================

# Global limits (reduced by 1/3 - will slow down automatically on 429/503 errors)
MAX_GLOBAL_CONCURRENT = 400      # Reduced by 1/3: Total concurrent requests - start fast, slow down if needed
MAX_PER_DOMAIN_CONCURRENT = 13   # Reduced by 1/3: Concurrent requests per domain - aggressive, adapts on errors

# Backoff settings
BACKOFF_PAUSE_SECONDS = 60       # Pause domain for 60s on 429/503
BACKOFF_SLOWDOWN_SECONDS = 5     # Permanent additional delay after backoff
MAX_RETRIES_PER_URL = 2          # Max retries for a single URL


class DomainStatus(Enum):
    """Status of a domain for rate limiting."""
    ACTIVE = "active"
    COOLING_DOWN = "cooling_down"
    SLOWED = "slowed"


@dataclass
class DomainState:
    """Tracks the state of a domain for adaptive rate limiting."""
    semaphore: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(MAX_PER_DOMAIN_CONCURRENT))
    status: DomainStatus = DomainStatus.ACTIVE
    last_request_time: float = 0.0
    cooldown_until: float = 0.0
    extra_delay: float = 0.0  # Permanent slowdown after backoff
    adaptive_delay: float = DELAY_BETWEEN_REQUESTS  # Dynamic delay that adapts to server behavior
    active_requests: int = 0
    total_requests: int = 0
    successful_requests: int = 0  # Track successful requests for adaptive delay
    errors_429: int = 0
    errors_503: int = 0
    other_errors: int = 0  # Track other errors (timeouts, connection errors, etc.)
    
    @property
    def is_cooling_down(self) -> bool:
        return time.time() < self.cooldown_until
    
    @property
    def current_delay(self) -> float:
        """Calculate current delay including adaptive and permanent slowdown."""
        return self.adaptive_delay + self.extra_delay
    
    def adjust_delay_on_success(self):
        """Gradually reduce delay when requests are successful."""
        # If we've had at least 10 successful requests, gradually reduce delay
        if self.successful_requests >= 10:
            # Reduce by 5% each time, but not below MIN_DELAY
            self.adaptive_delay = max(MIN_DELAY, self.adaptive_delay * 0.95)
    
    def adjust_delay_on_error(self):
        """Increase delay when errors occur (but not 429/503 which use backoff)."""
        # Increase by 50% on error, but not above MAX_DELAY
        self.adaptive_delay = min(MAX_DELAY, self.adaptive_delay * 1.5)


@dataclass
class CrawlStats:
    """Statistics for crawling progress."""
    sites_scraped: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    sentences_collected: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    failed_urls: int = 0
    skipped_duplicates: int = 0
    queue_size: int = 0
    active_workers: int = 0
    current_urls: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    total_runtime_seconds: float = 0.0  # Cumulative runtime across all runs
    
    # Adaptive rate limiting stats
    domains_cooling_down: List[str] = field(default_factory=list)
    domains_slowed: List[str] = field(default_factory=list)
    total_429_errors: int = 0
    total_503_errors: int = 0
    total_retries: int = 0
    
    # For time estimation
    initial_queue_size: int = 0
    
    @property
    def total_sites(self) -> int:
        return sum(self.sites_scraped.values())
    
    @property
    def total_sentences(self) -> int:
        return sum(self.sentences_collected.values())
    
    @property
    def elapsed_time(self) -> float:
        """Current session elapsed time."""
        return time.time() - self.start_time
    
    @property
    def total_elapsed_time(self) -> float:
        """Total elapsed time including previous runs."""
        return self.total_runtime_seconds + self.elapsed_time
    
    @property
    def urls_per_minute(self) -> float:
        elapsed = self.elapsed_time
        if elapsed > 0:
            return (self.total_sites / elapsed) * 60
        return 0
    
    @property
    def urls_per_second(self) -> float:
        elapsed = self.elapsed_time
        if elapsed > 0:
            return self.total_sites / elapsed
        return 0
    
    @property
    def estimated_remaining_seconds(self) -> float:
        """Estimate remaining time based on current speed."""
        if self.urls_per_second > 0 and self.queue_size > 0:
            return self.queue_size / self.urls_per_second
        return 0
    
    @property
    def estimated_total_seconds(self) -> float:
        """Estimate total time for the entire crawl."""
        return self.elapsed_time + self.estimated_remaining_seconds
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        total_processed = self.total_sites + self.failed_urls + self.skipped_duplicates
        total_work = total_processed + self.queue_size
        if total_work > 0:
            return (total_processed / total_work) * 100
        return 0


@dataclass
class CrawlTask:
    """A single crawl task."""
    url: str
    depth: int
    religion: str
    source: str
    region: str
    domain: str
    retry_count: int = 0


class DomainRotatingQueue:
    """
    A queue that rotates between domains to maximize parallelism.
    Instead of processing URLs in FIFO order, it ensures we're always
    hitting multiple domains simultaneously.
    
    Supports persistence to disk for resume functionality.
    """
    
    def __init__(self):
        self._domain_queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self._domains: List[str] = []
        self._current_idx = 0
        self._lock = asyncio.Lock()
        self._size = 0
    
    async def put(self, task: CrawlTask):
        """Add a task to its domain queue."""
        async with self._lock:
            domain = task.domain
            if domain not in self._domain_queues:
                self._domains.append(domain)
            await self._domain_queues[domain].put(task)
            self._size += 1
    
    async def get(self) -> CrawlTask:
        """Get a task, rotating between domains."""
        async with self._lock:
            if not self._domains:
                raise asyncio.QueueEmpty()
            
            # Try each domain in round-robin
            attempts = len(self._domains)
            for _ in range(attempts):
                self._current_idx = (self._current_idx + 1) % len(self._domains)
                domain = self._domains[self._current_idx]
                queue = self._domain_queues[domain]
                
                if not queue.empty():
                    task = await queue.get()
                    self._size -= 1
                    
                    # Remove empty domain queues
                    if queue.empty():
                        del self._domain_queues[domain]
                        self._domains.remove(domain)
                        if self._current_idx >= len(self._domains):
                            self._current_idx = 0
                    
                    return task
            
            raise asyncio.QueueEmpty()
    
    async def get_all_tasks(self) -> List[CrawlTask]:
        """Extract all tasks from the queue (for persistence)."""
        tasks = []
        temp_storage = {}  # domain -> list of tasks
        
        async with self._lock:
            # Extract all tasks from all domain queues
            for domain in list(self._domains):
                queue = self._domain_queues[domain]
                domain_tasks = []
                # Get all tasks from this domain's queue
                while True:
                    try:
                        task = queue.get_nowait()  # Non-blocking get
                        domain_tasks.append(task)
                    except asyncio.QueueEmpty:
                        break
                if domain_tasks:
                    temp_storage[domain] = domain_tasks
                    tasks.extend(domain_tasks)
            
            # Rebuild queues with extracted tasks
            self._domain_queues.clear()
            self._domains.clear()
            self._size = 0
            
            for domain, domain_tasks in temp_storage.items():
                self._domain_queues[domain] = asyncio.Queue()
                self._domains.append(domain)
                for task in domain_tasks:
                    await self._domain_queues[domain].put(task)
                    self._size += 1
        
        return tasks
    
    async def load_from_tasks(self, tasks: List[CrawlTask]):
        """Load tasks into the queue (for persistence)."""
        async with self._lock:
            # Clear existing queues
            self._domain_queues.clear()
            self._domains.clear()
            self._size = 0
            
            # Add all tasks back
            for task in tasks:
                domain = task.domain
                if domain not in self._domain_queues:
                    self._domain_queues[domain] = asyncio.Queue()
                    self._domains.append(domain)
                await self._domain_queues[domain].put(task)
                self._size += 1
    
    async def get_all_urls(self) -> Set[str]:
        """Get set of all URLs currently in the queue (for deduplication)."""
        tasks = await self.get_all_tasks()
        return {task.url for task in tasks}
    
    def qsize(self) -> int:
        return self._size
    
    def empty(self) -> bool:
        return self._size == 0
    
    def domain_count(self) -> int:
        return len(self._domains)


class AdaptiveRateLimiter:
    """
    Ethical, adaptive rate limiter that respects server capacity.
    
    - Limits concurrent requests globally (saves RAM)
    - Limits concurrent requests per domain (prevents server overload)
    - Backs off on 429/503 errors (respects server signals)
    - Permanently slows down problematic domains
    """
    
    def __init__(self):
        # Global semaphore to limit total concurrent requests
        self.global_semaphore = asyncio.Semaphore(MAX_GLOBAL_CONCURRENT)
        
        # Per-domain state tracking
        self.domain_states: Dict[str, DomainState] = {}
        self._lock = asyncio.Lock()
    
    def _get_domain_state(self, domain: str) -> DomainState:
        """Get or create domain state."""
        if domain not in self.domain_states:
            self.domain_states[domain] = DomainState()
        return self.domain_states[domain]
    
    async def acquire(self, domain: str) -> bool:
        """
        Acquire permission to make a request to a domain.
        Returns False if domain is cooling down.
        """
        state = self._get_domain_state(domain)
        
        # Check if domain is cooling down
        if state.is_cooling_down:
            return False
        
        # Acquire global semaphore first
        await self.global_semaphore.acquire()
        
        # Then acquire domain-specific semaphore
        await state.semaphore.acquire()
        
        # Apply rate limiting delay
        async with self._lock:
            elapsed = time.time() - state.last_request_time
            delay_needed = state.current_delay - elapsed
            
            if delay_needed > 0:
                await asyncio.sleep(delay_needed)
            
            state.last_request_time = time.time()
            state.active_requests += 1
            state.total_requests += 1
        
        return True
    
    async def release(self, domain: str):
        """Release the request slot for a domain."""
        state = self._get_domain_state(domain)
        
        async with self._lock:
            state.active_requests = max(0, state.active_requests - 1)
        
        state.semaphore.release()
        self.global_semaphore.release()
    
    async def report_error(self, domain: str, status_code: int):
        """
        Report an error for a domain. Triggers backoff on 429/503.
        For other errors, increases adaptive delay.
        """
        state = self._get_domain_state(domain)
        
        async with self._lock:
            if status_code == 429:
                state.errors_429 += 1
                state.status = DomainStatus.COOLING_DOWN
                state.cooldown_until = time.time() + BACKOFF_PAUSE_SECONDS
                state.extra_delay += BACKOFF_SLOWDOWN_SECONDS
                
            elif status_code == 503:
                state.errors_503 += 1
                state.status = DomainStatus.COOLING_DOWN
                state.cooldown_until = time.time() + BACKOFF_PAUSE_SECONDS
                state.extra_delay += BACKOFF_SLOWDOWN_SECONDS
            else:
                # Other errors (timeouts, connection errors, etc.) - increase adaptive delay
                state.other_errors += 1
                state.adjust_delay_on_error()
    
    async def report_success(self, domain: str):
        """Report a successful request. Clears cooling down status and adjusts delay."""
        state = self._get_domain_state(domain)
        
        async with self._lock:
            state.successful_requests += 1
            # Gradually reduce delay when things are working well
            state.adjust_delay_on_success()
            
            if state.status == DomainStatus.COOLING_DOWN and not state.is_cooling_down:
                # Cooling period is over, mark as slowed if we added extra delay
                if state.extra_delay > 0:
                    state.status = DomainStatus.SLOWED
                else:
                    state.status = DomainStatus.ACTIVE
    
    def get_cooling_down_domains(self) -> List[str]:
        """Get list of domains currently cooling down."""
        return [
            domain for domain, state in self.domain_states.items()
            if state.is_cooling_down
        ]
    
    def get_slowed_domains(self) -> List[str]:
        """Get list of domains that have been permanently slowed."""
        return [
            domain for domain, state in self.domain_states.items()
            if state.extra_delay > 0
        ]
    
    def get_domain_status(self, domain: str) -> Dict:
        """Get detailed status for a domain."""
        state = self._get_domain_state(domain)
        return {
            'status': state.status.value,
            'active_requests': state.active_requests,
            'total_requests': state.total_requests,
            'extra_delay': state.extra_delay,
            'errors_429': state.errors_429,
            'errors_503': state.errors_503,
            'cooling_down': state.is_cooling_down,
            'cooldown_remaining': max(0, state.cooldown_until - time.time()),
        }


class AsyncCrawler:
    """
    High-performance async web crawler with:
    - Ethical per-domain rate limiting (1 concurrent per domain)
    - Global concurrency limit (50 total to save RAM)
    - Adaptive backoff on 429/503 errors
    - Domain rotation for maximum parallelism
    - ProcessPoolExecutor for CPU-bound NLP
    - User-agent rotation
    - URL trap detection
    - robots.txt respect
    - Separate output files by religion
    """
    
    def __init__(self):
        self.visited: Set[str] = set()
        self.content_hashes: Set[str] = set()
        self.queue = DomainRotatingQueue()
        self.domain_counts: Dict[str, int] = defaultdict(int)
        
        # Adaptive rate limiter
        self.rate_limiter = AdaptiveRateLimiter()
        
        # Robots.txt cache
        self.robots_cache: Dict[str, RobotFileParser] = {}
        
        self.stats = CrawlStats()
        
        # Separate sentence storage by religion
        self.sentences_by_religion: Dict[str, List[Dict]] = defaultdict(list)
        self.rejected_sentences: List[Dict] = []  # Non-Indonesian sentences
        # Per-URL sentence deduplication (same sentence on same URL = duplicate)
        self.seen_sentences_per_url: Dict[str, Set[str]] = defaultdict(set)
        self.sentences_lock = asyncio.Lock()
        self._history_lock = asyncio.Lock()
        
        # Track URLs at MAX_DEPTH for continuation
        self.depth_boundary_urls: List[Dict] = []  # URLs discovered at MAX_DEPTH
        self._boundary_lock = asyncio.Lock()
        
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None
        
        # ProcessPoolExecutor for CPU-bound NLP work
        num_workers = NLP_WORKERS or max(1, (os.cpu_count() or 4) - 1)
        self._nlp_executor = ProcessPoolExecutor(max_workers=num_workers)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    async def initialize(self):
        """Initialize crawler state and session."""
        self._loop = asyncio.get_running_loop()
        
        await self._load_history()
        await self._load_content_hashes()
        await self._load_queue()  # Load persistent queue before loading seeds
        await self._load_depth_boundary_urls()  # Load boundary URLs
        await self._load_persistent_stats()  # Load persistent stats (runtime, etc.)
        await self._load_initial_stats()  # Load statistics from existing CSV files
        await self._load_seeds()
        
        # Optimized connection settings
        timeout = aiohttp.ClientTimeout(
            total=REQUEST_TIMEOUT,
            connect=5,
            sock_read=REQUEST_TIMEOUT
        )
        
        # Create SSL context - disable verification for maximum compatibility
        # Many Indonesian religious sites have SSL issues, and macOS Python often
        # has certificate problems. This ensures the scraper works reliably.
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        connector = aiohttp.TCPConnector(
            limit=CONNECTION_LIMIT,
            limit_per_host=CONNECTION_LIMIT_PER_HOST,
            ttl_dns_cache=DNS_CACHE_TTL,
            keepalive_timeout=KEEPALIVE_TIMEOUT,
            enable_cleanup_closed=True,
            force_close=False,
            ssl=ssl_context,
        )
        
        self._session = aiohttp.ClientSession(
            headers=REQUEST_HEADERS,
            timeout=timeout,
            connector=connector,
            cookie_jar=aiohttp.DummyCookieJar(),
        )
    
    async def close(self):
        """Close the session, executor, and save state."""
        if self._session:
            await self._session.close()
        
        self._nlp_executor.shutdown(wait=False)
        await self._save_content_hashes()
        await self._save_queue()  # Save queue state
        await self._save_depth_boundary_urls()  # Save depth boundary URLs
        await self._save_persistent_stats()  # Save persistent stats (runtime, etc.)
    
    async def _load_history(self):
        """Load previously visited URLs."""
        try:
            async with aiofiles.open(HISTORY_LOG, mode='r', encoding=FILE_ENCODING) as f:
                content = await f.read()
                for line in content.splitlines():
                    url = line.strip()
                    if url:
                        self.visited.add(url)
        except FileNotFoundError:
            pass
    
    async def _save_to_history(self, url: str):
        """Append URL to history log (thread-safe)."""
        async with self._history_lock:
            async with aiofiles.open(HISTORY_LOG, mode='a', encoding=FILE_ENCODING) as f:
                await f.write(url + '\n')
    
    async def _load_content_hashes(self):
        """Load content hashes for deduplication."""
        if not ENABLE_DEDUPLICATION:
            return
        
        try:
            async with aiofiles.open(CONTENT_HASHES_FILE, mode='r', encoding=FILE_ENCODING) as f:
                content = await f.read()
                for line in content.splitlines():
                    h = line.strip()
                    if h:
                        self.content_hashes.add(h)
        except FileNotFoundError:
            pass
    
    async def _save_content_hashes(self):
        """Save content hashes."""
        if not ENABLE_DEDUPLICATION or not self.content_hashes:
            return
        
        async with aiofiles.open(CONTENT_HASHES_FILE, mode='w', encoding=FILE_ENCODING) as f:
            await f.write('\n'.join(self.content_hashes))
    
    async def _load_queue(self):
        """Load queue from persistent storage."""
        try:
            async with aiofiles.open(QUEUE_FILE, mode='r', encoding=FILE_ENCODING) as f:
                content = await f.read()
                if content.strip():
                    tasks_data = json.loads(content)
                    tasks = [CrawlTask(**task) for task in tasks_data]
                    # Filter out tasks that are already visited
                    unvisited_tasks = [t for t in tasks if t.url not in self.visited]
                    if unvisited_tasks:
                        await self.queue.load_from_tasks(unvisited_tasks)
        except FileNotFoundError:
            pass
        except Exception:
            # If loading fails, just start fresh
            pass
    
    async def _save_queue(self):
        """Save queue to persistent storage."""
        try:
            # Get all tasks from queue
            tasks = await self.queue.get_all_tasks()
            # Convert to dict for JSON serialization
            tasks_data = [
                {
                    'url': task.url,
                    'depth': task.depth,
                    'religion': task.religion,
                    'source': task.source,
                    'region': task.region,
                    'domain': task.domain,
                    'retry_count': task.retry_count,
                }
                for task in tasks
            ]
            # Always save queue state (even if empty, so we know the state)
            async with aiofiles.open(QUEUE_FILE, mode='w', encoding=FILE_ENCODING) as f:
                await f.write(json.dumps(tasks_data, indent=2))
        except Exception:
            # If saving fails, don't crash
            pass
    
    async def _load_depth_boundary_urls(self):
        """Load depth boundary URLs from previous runs."""
        try:
            async with aiofiles.open(DEPTH_BOUNDARY_FILE, mode='r', encoding=FILE_ENCODING) as f:
                content = await f.read()
                if content.strip():
                    loaded_urls = json.loads(content)
                    # Filter out already visited URLs on load
                    self.depth_boundary_urls = [
                        entry for entry in loaded_urls 
                        if entry.get('url') not in self.visited
                    ]
        except FileNotFoundError:
            self.depth_boundary_urls = []
        except Exception:
            self.depth_boundary_urls = []
    
    async def _save_depth_boundary_urls(self):
        """Save URLs at MAX_DEPTH for continuation."""
        try:
            async with self._boundary_lock:
                # Deduplicate by URL
                seen = set()
                unique_urls = []
                for entry in self.depth_boundary_urls:
                    url = entry['url']
                    if url not in seen and url not in self.visited:
                        seen.add(url)
                        unique_urls.append(entry)
                
                if unique_urls:
                    async with aiofiles.open(DEPTH_BOUNDARY_FILE, mode='w', encoding=FILE_ENCODING) as f:
                        await f.write(json.dumps(unique_urls, indent=2, ensure_ascii=False))
        except Exception:
            # If saving fails, don't crash
            pass
    
    async def _load_initial_stats(self):
        """Load initial statistics from existing CSV files."""
        import csv as csv_module
        
        # Map religion names to CSV file names
        csv_files = {
            'Catholic': BASE_DIR / 'data' / 'scraped' / 'Scraped_Catholic.csv',
            'Islam': BASE_DIR / 'data' / 'scraped' / 'Scraped_Islam.csv',
            'Protestant': BASE_DIR / 'data' / 'scraped' / 'Scraped_Protestant.csv',
        }
        
        for religion, csv_path in csv_files.items():
            if not csv_path.exists():
                continue
            
            try:
                # Read CSV file to count sentences and unique sites
                sentence_count = 0
                unique_sites = set()
                
                async with aiofiles.open(csv_path, mode='r', encoding=FILE_ENCODING) as f:
                    content = await f.read()
                    reader = csv_module.DictReader(io.StringIO(content))
                    
                    for row in reader:
                        # Only count rows with actual sentence content
                        sentence = row.get('Sentence_Unit', '').strip()
                        if sentence:
                            sentence_count += 1
                            url = row.get('Link', '').strip()
                            if url:
                                unique_sites.add(url)
                
                # Update stats with loaded data (additive, so new scrapes will increment)
                if sentence_count > 0:
                    self.stats.sentences_collected[religion] = sentence_count
                if unique_sites:
                    self.stats.sites_scraped[religion] = len(unique_sites)
                    
            except Exception:
                # If reading fails, just continue without loading stats
                pass
    
    async def _load_persistent_stats(self):
        """Load persistent statistics (runtime, etc.) from file."""
        try:
            if STATS_FILE.exists():
                async with aiofiles.open(STATS_FILE, mode='r', encoding=FILE_ENCODING) as f:
                    content = await f.read()
                    if content.strip():
                        stats_data = json.loads(content)
                        # Load total runtime from previous runs
                        self.stats.total_runtime_seconds = stats_data.get('total_runtime_seconds', 0.0)
        except Exception:
            # If loading fails, just start with 0
            self.stats.total_runtime_seconds = 0.0
    
    async def _save_persistent_stats(self):
        """Save persistent statistics (runtime, etc.) to file."""
        try:
            # Calculate total runtime including current session
            total_runtime = self.stats.total_runtime_seconds + self.stats.elapsed_time
            
            stats_data = {
                'total_runtime_seconds': total_runtime,
            }
            
            async with aiofiles.open(STATS_FILE, mode='w', encoding=FILE_ENCODING) as f:
                await f.write(json.dumps(stats_data, indent=2))
        except Exception:
            # If saving fails, don't crash
            pass
    
    async def _load_seeds(self):
        """Load seed URLs from JSON file (only if not already visited or queued)."""
        async with aiofiles.open(SEEDS_FILE, mode='r', encoding=FILE_ENCODING) as f:
            content = await f.read()
            seeds_data = json.loads(content)
        
        # Get URLs already in queue to avoid duplicates
        queued_urls = await self.queue.get_all_urls()
        
        for religion, sites in seeds_data.items():
            religion_cap = religion.capitalize()
            for site in sites:
                url = site['url']
                # Only add if not visited AND not already in queue
                if url not in self.visited and url not in queued_urls:
                    domain = urlparse(url).netloc
                    task = CrawlTask(
                        url=url,
                        depth=0,
                        religion=religion_cap,
                        source=site.get('source', 'Unknown'),
                        region=site.get('region', 'Unknown'),
                        domain=domain
                    )
                    await self.queue.put(task)
        
        self.stats.queue_size = self.queue.qsize()
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        url = url.split('#')[0]
        url = url.rstrip('/')
        url = re.sub(r'[?&](utm_\w+|ref|source|fbclid|gclid|_ga)=[^&]*', '', url)
        return url
    
    def _is_same_domain(self, url: str, base_domain: str) -> bool:
        """Check if URL belongs to the same domain."""
        url_domain = urlparse(url).netloc
        return url_domain == base_domain or url_domain.endswith('.' + base_domain)
    
    def _extract_links(self, html: str, base_url: str, base_domain: str) -> List[str]:
        """Extract same-domain links from HTML using regex (faster than parsing)."""
        links = []
        href_pattern = re.compile(r'href=["\']([^"\']+)["\']', re.I)
        
        for match in href_pattern.finditer(html):
            href = match.group(1)
            
            if href.startswith(('mailto:', 'tel:', 'javascript:', '#', 'data:')):
                continue
            
            full_url = urljoin(base_url, href)
            full_url = self._normalize_url(full_url)
            
            if not self._is_same_domain(full_url, base_domain):
                continue
            
            # Skip URL traps
            if is_url_trap(full_url):
                continue
            
            links.append(full_url)
        
        return list(set(links))
    
    async def _extract_sentences_async(self, html: str) -> List[dict]:
        """Run CPU-bound NLP extraction in ProcessPoolExecutor."""
        from nlp_processor import extract_sentences
        
        try:
            results = await self._loop.run_in_executor(
                self._nlp_executor,
                extract_sentences,
                html
            )
            return results
        except Exception:
            return extract_sentences(html)
    
    async def _check_robots_txt(self, url: str, domain: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        if domain not in self.robots_cache:
            try:
                robots_url = f"https://{domain}/robots.txt"
                rp = RobotFileParser()
                rp.set_url(robots_url)
                
                # Fetch robots.txt
                async with self._session.get(robots_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        content = await resp.text()
                        rp.parse(content.splitlines())
                    else:
                        # No robots.txt = allow all
                        rp.parse([])
                
                self.robots_cache[domain] = rp
            except Exception:
                # If we can't fetch robots.txt, allow
                self.robots_cache[domain] = RobotFileParser()
                self.robots_cache[domain].parse([])
        
        return self.robots_cache[domain].can_fetch("*", url)
    
    async def _fetch_page(self, url: str, domain: str) -> Tuple[Optional[str], int]:
        """
        Fetch page with adaptive rate limiting and user-agent rotation.
        Returns (html_content, status_code).
        """
        # Check robots.txt first
        if not await self._check_robots_txt(url, domain):
            return None, 0  # Blocked by robots.txt
        
        # Try to acquire rate limiter
        acquired = await self.rate_limiter.acquire(domain)
        if not acquired:
            # Domain is cooling down
            return None, 0
        
        try:
            # Use rotating user agent
            headers = {'User-Agent': get_random_user_agent()}
            
            async with self._session.get(url, headers=headers, allow_redirects=True) as response:
                status = response.status
                
                # Handle rate limiting errors
                if status in (429, 503):
                    await self.rate_limiter.report_error(domain, status)
                    if status == 429:
                        self.stats.total_429_errors += 1
                    else:
                        self.stats.total_503_errors += 1
                    return None, status
                
                if status != 200:
                    # Report 4xx/5xx errors (but not 429/503 which are handled above)
                    if 400 <= status < 600 and status not in (429, 503):
                        await self.rate_limiter.report_error(domain, status)
                    return None, status
                
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' not in content_type:
                    return None, status
                
                html = await response.text()
                await self.rate_limiter.report_success(domain)
                return html, status
                
        except Exception:
            # Timeouts, connection errors, etc. - increase adaptive delay
            await self.rate_limiter.report_error(domain, 0)  # 0 = generic error
            return None, 0
        finally:
            await self.rate_limiter.release(domain)
    
    def _update_rate_limiter_stats(self):
        """Update stats with rate limiter information."""
        self.stats.domains_cooling_down = self.rate_limiter.get_cooling_down_domains()
        self.stats.domains_slowed = self.rate_limiter.get_slowed_domains()
    
    async def _process_task(self, task: CrawlTask):
        """Process a single crawl task with retry logic."""
        url = task.url
        
        # Skip if already visited
        if url in self.visited:
            return
        
        # Skip if domain limit reached
        if MAX_PAGES_PER_DOMAIN > 0 and self.domain_counts[task.domain] >= MAX_PAGES_PER_DOMAIN:
            return
        
        # Check if domain is cooling down - requeue if so
        if task.domain in self.stats.domains_cooling_down:
            if task.retry_count < MAX_RETRIES_PER_URL:
                task.retry_count += 1
                await self.queue.put(task)
                self.stats.total_retries += 1
            return
        
        # Mark as visited
        self.visited.add(url)
        await self._save_to_history(url)
        
        # Update dashboard
        self.stats.current_urls.append(url[-60:])
        if len(self.stats.current_urls) > 8:
            self.stats.current_urls.pop(0)
        
        # Fetch page
        html, status = await self._fetch_page(url, task.domain)
        
        # Update rate limiter stats
        self._update_rate_limiter_stats()
        
        # Handle retry on rate limit
        if status in (429, 503) and task.retry_count < MAX_RETRIES_PER_URL:
            # Remove from visited so it can be retried
            self.visited.discard(url)
            task.retry_count += 1
            await self.queue.put(task)
            self.stats.total_retries += 1
            return
        
        if not html or len(html) < MIN_CONTENT_LENGTH:
            self.stats.failed_urls += 1
            return
        
        # Content deduplication
        if ENABLE_DEDUPLICATION:
            content_hash = hash_content(html)
            if content_hash in self.content_hashes:
                self.stats.skipped_duplicates += 1
                return
            self.content_hashes.add(content_hash)
        
        # Extract sentences (returns list of dicts with 'sentence', 'is_indonesian', 'detected_lang')
        sentence_results = await self._extract_sentences_async(html)
        
        # Store sentences by religion and track rejected (with per-URL deduplication)
        async with self.sentences_lock:
            indonesian_count = 0
            for result in sentence_results:
                sentence_text = result['sentence']
                
                # Skip duplicate sentences on the SAME URL only
                # (same sentence on different URLs is allowed)
                if sentence_text in self.seen_sentences_per_url[url]:
                    continue
                self.seen_sentences_per_url[url].add(sentence_text)
                
                sentence_data = {
                    'sentence': sentence_text,
                    'religion': task.religion,
                    'source': task.source,
                    'region': task.region,
                    'url': url,
                    'detected_lang': result.get('detected_lang', 'unknown'),
                }
                
                if result.get('is_indonesian', True):
                    self.sentences_by_religion[task.religion].append(sentence_data)
                    indonesian_count += 1
                else:
                    # Store rejected (non-Indonesian) sentences for debugging
                    self.rejected_sentences.append(sentence_data)
            
            self.stats.sentences_collected[task.religion] += indonesian_count
        
        # Update stats
        self.stats.sites_scraped[task.religion] += 1
        self.domain_counts[task.domain] += 1
        
        # Queue new links if within depth limit
        if task.depth < MAX_DEPTH:
            new_links = self._extract_links(html, url, task.domain)
            for link in new_links:
                if link not in self.visited:
                    # Check if this link would be at MAX_DEPTH
                    if task.depth + 1 == MAX_DEPTH:
                        # Save to boundary URLs for continuation
                        async with self._boundary_lock:
                            # Check if URL already in boundary list (in-memory dedup)
                            if not any(entry.get('url') == link for entry in self.depth_boundary_urls):
                                self.depth_boundary_urls.append({
                                    'url': link,
                                    'religion': task.religion,
                                    'source': task.source,
                                    'region': task.region,
                                    'domain': task.domain,
                                    'discovered_from': url,
                                })
                    
                    new_task = CrawlTask(
                        url=link,
                        depth=task.depth + 1,
                        religion=task.religion,
                        source=task.source,
                        region=task.region,
                        domain=task.domain
                    )
                    await self.queue.put(new_task)
        
        self.stats.queue_size = self.queue.qsize()
        
        # Periodically save queue and boundary URLs (every 100 URLs)
        if self.stats.total_sites % 100 == 0:
            await self._save_queue()
            await self._save_depth_boundary_urls()
    
    async def _worker(self, worker_id: int):
        """Worker coroutine that processes tasks from the queue."""
        while self._running:
            try:
                task = await asyncio.wait_for(
                    self._get_next_task(),
                    timeout=3.0
                )
                self.stats.active_workers += 1
                
                try:
                    await self._process_task(task)
                finally:
                    self.stats.active_workers -= 1
                    
            except asyncio.TimeoutError:
                if self.queue.empty():
                    break
            except asyncio.CancelledError:
                break
    
    async def _get_next_task(self) -> CrawlTask:
        """Get next task with retry."""
        while True:
            try:
                return await self.queue.get()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.1)
                if self.queue.empty():
                    raise asyncio.TimeoutError()
    
    async def get_sentences_batch(self) -> Tuple[Dict[str, List[Dict]], List[Dict]]:
        """
        Get and clear collected sentences.
        
        Returns:
            Tuple of (sentences_by_religion dict, rejected_sentences list)
        """
        async with self.sentences_lock:
            by_religion = {k: v.copy() for k, v in self.sentences_by_religion.items()}
            rejected = self.rejected_sentences.copy()
            
            # Clear
            for key in self.sentences_by_religion:
                self.sentences_by_religion[key].clear()
            self.rejected_sentences.clear()
            
            return by_religion, rejected
    
    async def run(self, num_workers: int = MAX_CONCURRENT_REQUESTS):
        """Run the crawler with multiple workers."""
        self._running = True
        
        # Cap workers at global limit
        num_workers = min(num_workers, MAX_GLOBAL_CONCURRENT)
        
        # Start workers
        workers = [asyncio.create_task(self._worker(i)) for i in range(num_workers)]
        
        # Wait for completion
        try:
            await asyncio.gather(*workers, return_exceptions=True)
        except asyncio.CancelledError:
            pass
        
        self._running = False
    
    def stop(self):
        """Signal the crawler to stop."""
        self._running = False
    
    def get_domain_status(self, domain: str) -> Dict:
        """Get detailed status for a specific domain."""
        return self.rate_limiter.get_domain_status(domain)
