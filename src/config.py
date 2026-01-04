"""
Configuration settings for the Religious Text Scraper & NLP Pipeline.
"""

from pathlib import Path

# =============================================================================
# PATHS (using pathlib for cross-platform compatibility)
# =============================================================================

BASE_DIR = Path(__file__).parent.parent.resolve()  # Go up one level from src/ to project root
SEEDS_FILE = BASE_DIR / "configs" / "seeds.json"
HISTORY_LOG = BASE_DIR / "data" / "crawler_state" / "history.log"
OUTPUT_CSV = BASE_DIR / "data" / "combined" / "religious_corpus.csv"
STATE_FILE = BASE_DIR / "data" / "crawler_state" / "crawler_state.json"
CONTENT_HASHES_FILE = BASE_DIR / "data" / "crawler_state" / "content_hashes.txt"
QUEUE_FILE = BASE_DIR / "data" / "crawler_state" / "queue.json"  # Persistent queue storage
DEPTH_BOUNDARY_FILE = BASE_DIR / "data" / "crawler_state" / "depth_boundary_urls.json"  # URLs at MAX_DEPTH for continuation
STATS_FILE = BASE_DIR / "data" / "crawler_state" / "crawler_stats.json"  # Persistent statistics (runtime, etc.)
FASTTEXT_MODEL = BASE_DIR / "src" / "lid.176.ftz"  # Keep model in src/ if it exists

# =============================================================================
# CRAWLING SETTINGS (OPTIMIZED)
# =============================================================================

MAX_DEPTH = 8                    # Maximum depth to crawl from seed URLs
MAX_CONCURRENT_REQUESTS = 267    # Reduced by 1/3: Aggressive parallel requests - will slow down if servers complain
MAX_CONCURRENT_PER_DOMAIN = 5    # Limit per single domain to avoid blocks (kept conservative)
REQUEST_TIMEOUT = 10             # Lower timeout for faster failures
DELAY_BETWEEN_REQUESTS = 0.01875 # Slowed by 1/3: Starting delay (adaptive - decreases on success, increases on errors)
MIN_DELAY = 0.00375              # Slowed by 1/3: Minimum delay when everything is working well
MAX_DELAY = 2.0                  # Maximum adaptive delay before backoff kicks in
SAVE_INTERVAL = 100              # Save to CSV every N URLs
MAX_PAGES_PER_DOMAIN = 100000    # Limit pages per domain (0 for unlimited)

# Connection pool settings (reduced by 1/3)
CONNECTION_LIMIT = 533           # Reduced by 1/3: Total connection pool size
CONNECTION_LIMIT_PER_HOST = 53   # Reduced by 1/3: Connections per host
DNS_CACHE_TTL = 300              # DNS cache time-to-live (seconds)
KEEPALIVE_TIMEOUT = 30           # Keep connections alive

# =============================================================================
# HTTP HEADERS
# =============================================================================

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# =============================================================================
# NLP SETTINGS
# =============================================================================

MIN_SENTENCE_LENGTH = 20
MAX_SENTENCE_LENGTH = 1000
MIN_WORDS_PER_SENTENCE = 3
INDONESIAN_CONFIDENCE_THRESHOLD = 2  # Minimum Indonesian markers for mixed text

# Content deduplication
ENABLE_DEDUPLICATION = True      # Skip pages with duplicate content
MIN_CONTENT_LENGTH = 100         # Minimum content length to process

# =============================================================================
# MULTIPROCESSING (CPU-BOUND NLP)
# =============================================================================

# Number of CPU workers for NLP processing (None = auto-detect)
NLP_WORKERS = None  # Will use os.cpu_count() if None

# =============================================================================
# FILE ENCODING (Cross-platform)
# =============================================================================

FILE_ENCODING = "utf-8"
CSV_NEWLINE = ""  # Required for proper CSV handling across platforms
