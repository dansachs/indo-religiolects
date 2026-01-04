"""
NLP Processing Pipeline - Master Specification Implementation
=============================================================

This module implements the complete text processing pipeline with:
1. Critical library management & notifications
2. Strict language detection (langdetect only)
3. Text cleaning & pre-processing pipeline
4. Structural noise filtering (sentence level)
5. Arabic script removal
"""

import re
import sys
import unicodedata
from typing import List, Optional, Tuple

# =============================================================================
# 1. CRITICAL LIBRARY MANAGEMENT & NOTIFICATIONS
# =============================================================================

def check_critical_dependencies():
    """
    Check for required libraries at startup.
    Exit immediately with loud error if any are missing.
    """
    missing = []
    
    # Check langdetect
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0  # For reproducibility
    except ImportError:
        missing.append("langdetect")
    
    # Check nltk
    try:
        import nltk
    except ImportError:
        missing.append("nltk")
    
    # Check trafilatura
    try:
        import trafilatura
    except ImportError:
        missing.append("trafilatura")
    
    # Check ftfy (optional but recommended)
    try:
        import ftfy
    except ImportError:
        print("[WARNING] ftfy not installed. Mojibake fixing will use manual fallback.")
        print("         Install with: pip install ftfy")
    
    if missing:
        print("\n" + "=" * 60)
        print("[CRITICAL ERROR] Missing required libraries!")
        print("=" * 60)
        for lib in missing:
            print(f"  ✗ {lib}")
        print("\nInstall missing libraries with:")
        print(f"  pip install {' '.join(missing)}")
        print("=" * 60 + "\n")
        sys.exit(1)
    
    # Check NLTK data
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            print("[INFO] Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to download NLTK data: {e}")
            sys.exit(1)
    
    try:
        import nltk
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except Exception:
            pass  # punkt_tab is optional


# Run dependency check on module import
check_critical_dependencies()

# Now safe to import
import nltk
from langdetect import detect, LangDetectException

# Try to import ftfy (optional)
try:
    import ftfy
    FTFY_AVAILABLE = True
except ImportError:
    FTFY_AVAILABLE = False


# =============================================================================
# 2. LANGUAGE DETECTION (STRICT)
# =============================================================================

def is_indonesian_robust(text: str) -> Tuple[bool, str]:
    """
    Strict language detection using langdetect ONLY.
    
    Returns:
        Tuple of (is_indonesian: bool, detected_language: str)
    
    Logic:
        - Return True ONLY if detected language is 'id', 'ms', or 'tl' (Malay family)
        - Mixed content (Indonesian with some English words) is allowed if langdetect classifies it as Indonesian/Malay
        - Return False for all other languages (including English, African languages, etc.)
        - Return False if text length < 15 characters
    """
    if not text or len(text.strip()) < 15:
        return False, "too_short"
    
    try:
        detected = detect(text)
        
        # Accept ONLY Indonesian, Malay, Tagalog (Malay family roots)
        if detected in ('id', 'ms', 'tl'):
            return True, detected
        
        # Reject all other languages (English, African languages, etc.)
        return False, detected
        
    except LangDetectException:
        # If detection fails, reject to be safe (user wants only Indonesian/Malay)
        return False, "detection_failed"


# =============================================================================
# 3. TEXT CLEANING & PRE-PROCESSING PIPELINE
# =============================================================================

# 3.1 Mojibake replacements (manual fallback)
MOJIBAKE_REPLACEMENTS = {
    'â€"': '-',
    'â€"': '—',
    'â€˜': "'",
    'â€™': "'",
    'â€œ': '"',
    'â€': '"',
    'â€¦': '...',
    '&nbsp;': ' ',
    'Â ': ' ',
    'Ã¢': 'â',
    'Ã©': 'é',
    'Ã¨': 'è',
    'Ã': 'à',
    '\x00': '',  # Null bytes
    '\ufeff': '',  # BOM
}

# 3.2 Title protection patterns (prevent bad sentence splitting)
TITLE_PATTERNS = [
    (r'\bBp\.', 'Bp'),
    (r'\bMgr\.', 'Mgr'),
    (r'\bPdt\.', 'Pdt'),
    (r'\bRm\.', 'Rm'),
    (r'\bSr\.', 'Sr'),
    (r'\bFr\.', 'Fr'),
    (r'\bSt\.', 'St'),
    (r'\bDrs\.', 'Drs'),
    (r'\bDr\.', 'Dr'),
    (r'\bProf\.', 'Prof'),
    (r'\bKH\.', 'KH'),
    (r'\bH\.', 'H'),
    (r'\bHj\.', 'Hj'),
    (r'\bUst\.', 'Ust'),
    (r'\bUstad\.', 'Ustad'),
    (r'\bUstadz\.', 'Ustadz'),
    (r'\bRev\.', 'Rev'),
    (r'\bPastor\.', 'Pastor'),
]

# 3.3 Navigation/Sidebar keywords to strip
NAV_KEYWORDS = [
    'facebook', 'instagram', 'twitter', 'youtube', 'whatsapp', 'telegram',
    'tiktok', 'home', 'menu', 'beranda', 'tentang', 'download', 'kategori',
    'arsip', 'komentar', 'baca juga', 'read more', 'selengkapnya',
    'lihat semua', 'view all', 'tags', 'share', 'bagikan', 'print',
    'cetak', 'next', 'previous', 'older', 'newer', 'subscribe',
    'newsletter', 'search', 'cari', 'login', 'logout', 'register',
    'daftar', 'sign in', 'sign up', 'follow us', 'ikuti kami',
    'artikel terkini', 'video terkini', 'berita terkini', 'post terkini',
    'artikel terbaru', 'video terbaru', 'berita terbaru',
]

# Patterns that indicate a sentence is just boilerplate navigation
BOILERPLATE_START_PATTERNS = [
    r'^artikel\s*/?\s*video\s+terkini',
    r'^baca\s+juga\s*:?',
    r'^lihat\s+juga\s*:?',
    r'^berita\s+terkait',
    r'^artikel\s+terkait',
    r'^tags?\s*:',
    r'^kategori\s*:',
    r'^share\s*:',
    r'^bagikan\s*:',
    r'^sumber\s*:',
    r'^foto\s*:',
    r'^credit\s*:',
    r'^gambar\s*:',
    r'^\d+\s*(shares?|likes?|views?|komentar)',
    r'^streaming\s+(tidak\s+)?terdengar',
    r'^mau\s+bertanya',
    r'^ingin\s+tahu\s+jawabannya',
    r'^kalo\s+aku\s+sih',
    r'^jika\s+streaming',
]

# 3.4 Bible reference pattern
BIBLE_REF_PATTERN = re.compile(
    r'\b(?:'
    r'(?:1|2|3|I{1,3})?\s*'
    r'(?:Yohanes|Matius|Markus|Lukas|Kisah|Roma|Korintus|Galatia|Efesus|'
    r'Filipi|Kolose|Tesalonika|Timotius|Titus|Filemon|Ibrani|Yakobus|'
    r'Petrus|Yudas|Wahyu|Kejadian|Keluaran|Imamat|Bilangan|Ulangan|'
    r'Yosua|Hakim|Rut|Samuel|Raja|Tawarikh|Ezra|Nehemia|Ester|Ayub|'
    r'Mazmur|Amsal|Pengkhotbah|Kidung|Yesaya|Yeremia|Ratapan|Yehezkiel|'
    r'Daniel|Hosea|Yoel|Amos|Obaja|Yunus|Mikha|Nahum|Habakuk|Zefanya|'
    r'Hagai|Zakharia|Maleakhi|'
    r'John|Matthew|Mark|Luke|Acts|Romans|Corinthians|Galatians|Ephesians|'
    r'Philippians|Colossians|Thessalonians|Timothy|Titus|Philemon|Hebrews|'
    r'James|Peter|Jude|Revelation|Genesis|Exodus|Leviticus|Numbers|'
    r'Deuteronomy|Joshua|Judges|Ruth|Samuel|Kings|Chronicles|Ezra|'
    r'Nehemiah|Esther|Job|Psalms?|Proverbs|Ecclesiastes|Song|Isaiah|'
    r'Jeremiah|Lamentations|Ezekiel|Daniel|Hosea|Joel|Amos|Obadiah|'
    r'Jonah|Micah|Nahum|Habakkuk|Zephaniah|Haggai|Zechariah|Malachi'
    r')\s*'
    r'\d{1,3}(?::\d{1,3}(?:\s*-\s*\d{1,3})?)?'
    r')\b',
    re.IGNORECASE
)

# Quran reference pattern
QURAN_REF_PATTERN = re.compile(
    r'\b(?:Q\.?S\.?|Surah?|Al-?)\s*[A-Za-z\-]+\s*(?:\(?\d{1,3}\)?)?(?::\d{1,3}(?:-\d{1,3})?)?\b',
    re.IGNORECASE
)

# 3.5 Arabic script pattern (to remove)
ARABIC_SCRIPT_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+')

# 3.6 Metadata patterns
METADATA_PATTERNS = [
    re.compile(r'\|\s*Halaman ini adalah.*', re.IGNORECASE),
    re.compile(r'©\s*\d{4}.*', re.IGNORECASE),
    re.compile(r'Copyright\s*©?.*', re.IGNORECASE),
    re.compile(r'All rights reserved.*', re.IGNORECASE),
    re.compile(r'Hak cipta.*', re.IGNORECASE),
]

# =============================================================================
# 3.7 DROP CAP / MISSING LETTER FIXES (Safety Net)
# =============================================================================
# 
# NOTE: The primary fix happens at HTML level in fix_drop_caps_html().
# This is just a minimal safety net for any that slip through.

# Only the most critical/common patterns that we've seen in the data
DROP_CAP_SAFETY_NET = {
    # Islamic terms (very common on Indonesian Islamic sites)
    r'\bsjid\b': 'Masjid',
    r'\bsyarakat\b': 'Masyarakat',
    r'\bikmah\b': 'Hikmah',
    r'\bkmurkan\b': 'Makmurkan',
    
    # Catholic/Christian terms
    r'\bomili\b': 'Homili',
    r'\bnjil\b': 'Injil',
    r'\bukaristi\b': 'Eukaristi',
    
    # Common Indonesian words that start sentences
    r'\bati\b': 'Hati',
    r'\badir\b': 'Hadir',
    r'\bari\b': 'Hari',
    r'\barapan\b': 'Harapan',
    r'\besar\b': 'Besar',
    r'\berita\b': 'Berita',
}

# Compile patterns (case-insensitive for flexibility)
_DROP_CAP_SAFETY_COMPILED = [
    (re.compile(p, re.IGNORECASE), r) for p, r in DROP_CAP_SAFETY_NET.items()
]


def fix_common_typos(text: str) -> str:
    """
    Safety net for drop cap fixes that slip through HTML-level processing.
    The main fix happens in fix_drop_caps_html() before text extraction.
    """
    if not text:
        return ""
    
    for pattern, replacement in _DROP_CAP_SAFETY_COMPILED:
        text = pattern.sub(replacement, text)
    
    return text


# =============================================================================
# 3.8 ADDRESS/PHONE JUNK DETECTION
# =============================================================================

# Address patterns (Indonesian) - expanded
ADDRESS_PATTERN = re.compile(
    r'(Jl\.|Jln\.|Jalan|Kotak Pos|Kode Pos|Kantor)\s+[A-Za-z0-9\s\.\,]+(\d{5}|Bandung|Jakarta|Surabaya|'
    r'Semarang|Yogyakarta|Medan|Makassar|Palembang|Denpasar|Jawa|Sumatera|Kalimantan|Sulawesi)',
    re.IGNORECASE
)

# Phone/Fax patterns - expanded
PHONE_PATTERN = re.compile(
    r'(Telp|Telepon|Fax|Faks|Hotline|Hubungi|No\.?\s*HP|WhatsApp|WA|Kotak Pos)\s*[:\.]?\s*'
    r'[\d\-\(\)\s\+]{5,}',
    re.IGNORECASE
)

# Standalone phone number pattern (just digits in phone format)
PHONE_NUMBER_PATTERN = re.compile(r'\(\d{3,4}\)\s*\d{4,}')

# Email pattern
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

# Postal code pattern
POSTAL_CODE_PATTERN = re.compile(r'\b\d{5}\b')

# Date concatenation pattern (date stuck to other text)
DATE_CONCAT_PATTERN = re.compile(
    r'(\d{1,2}\s*(Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember|'
    r'January|February|March|April|May|June|July|August|September|October|November|December)'
    r'\s*,?\s*\d{4})[A-Z]',
    re.IGNORECASE
)

# CamelCase concatenation (words stuck together without spaces)
CAMELCASE_CONCAT_PATTERN = re.compile(r'[a-z][A-Z][a-z]+[A-Z]')


def is_address_junk(text: str) -> bool:
    """
    Detect if a sentence is primarily contact info (address, phone, email).
    
    Returns True if the sentence should be rejected as "junk" contact data.
    """
    if not text:
        return False
    
    # Check for address patterns
    if ADDRESS_PATTERN.search(text):
        return True
    
    # Check for phone/fax patterns
    if PHONE_PATTERN.search(text):
        return True
    
    # Check for standalone phone numbers
    if PHONE_NUMBER_PATTERN.search(text):
        return True
    
    # Check for postal codes in short text
    words = text.split()
    if len(words) < 15 and POSTAL_CODE_PATTERN.search(text):
        return True
    
    # Check if mostly numbers and proper nouns (contact lists)
    if len(words) >= 3:
        digit_heavy = sum(1 for w in words if any(c.isdigit() for c in w))
        if digit_heavy / len(words) > 0.4:  # Lowered threshold
            return True
    
    # Check for email addresses in short sentences
    if len(words) < 10 and EMAIL_PATTERN.search(text):
        return True
    
    return False


def has_bad_concatenation(text: str) -> bool:
    """
    Detect if text has bad concatenation (words/dates stuck together).
    
    Examples:
    - "August 25, 2016Campaign Video" - date stuck to text
    - "UtaraKota" - CamelCase concatenation
    - "terkiniPOST" - lowercase followed by ALL CAPS
    - "Sekali..Bacaan" - double dots without space
    - "WHAT WE BELIEVEKREDOQ&A" - ALL CAPS navigation junk
    """
    if not text:
        return False
    
    # Check for date concatenation
    if DATE_CONCAT_PATTERN.search(text):
        return True
    
    # Check for lowercase followed by ALL CAPS word (e.g., "terkiniPOST")
    if re.search(r'[a-z]{2,}[A-Z]{3,}', text):
        return True
    
    # Check for double dots followed by letter (with or without space)
    # e.g., "kalo.. Gimana" or "Sekali..Bacaan"
    if re.search(r'\.\.\s*[A-Z]', text):
        return True
    
    # ALL CAPS words concatenated together (navigation junk)
    # e.g., "WHATWEBELIEVE", "KREDOQ&A"
    if re.search(r'[A-Z]{4,}[A-Z]{4,}', text):
        return True
    
    # Multiple ALL CAPS words in a row (3+ words) usually indicates navigation
    caps_word_count = len(re.findall(r'\b[A-Z]{3,}\b', text))
    if caps_word_count >= 3:
        return True
    
    # Check for excessive CamelCase concatenation (more than 2 instances)
    camelcase_matches = len(CAMELCASE_CONCAT_PATTERN.findall(text))
    if camelcase_matches >= 2:
        return True
    
    # Check for title-body concatenation patterns
    # e.g., "ManusiaAda", "BiaraPengantar", "ImamKodrat"
    # These happen when HTML headings merge with body text
    # Look for pattern: 2+ lowercase letters followed by uppercase + 2+ lowercase
    concat_matches = re.findall(r'[a-z]{2,}[A-Z][a-z]{2,}', text)
    if concat_matches:
        return True
    
    # ALL CAPS word stuck to next word: "KATOLIKANAJika" 
    # Pattern: 3+ uppercase followed by uppercase then 2+ lowercase
    caps_stuck = re.search(r'[A-Z]{3,}[A-Z][a-z]{2,}', text)
    if caps_stuck:
        return True
    
    # Sentences containing common navigation ALL CAPS terms
    nav_caps = ['MENU', 'HOME', 'KREDO', 'LITURGY', 'WHAT WE', 'SEE OTHERS', 'Q&A']
    text_upper = text.upper()
    if sum(1 for nav in nav_caps if nav in text_upper) >= 2:
        return True
    
    return False


def has_weird_ending(text: str) -> bool:
    """
    Detect sentences with weird/incomplete endings.
    
    Examples:
    - "...paroki.6." - ends with number
    - "...dan 3." - ends with just a number  
    - "...Berkoordinasi dengan" - incomplete phrase
    """
    if not text:
        return False
    
    text = text.strip()
    
    # Ends with a number followed by period: ".6." or "3."
    if re.search(r'\.\d+\.?$', text):
        return True
    
    # Ends with just a number
    if re.search(r'\s\d{1,2}\.?$', text):
        return True
    
    # Ends with common incomplete words or phrases
    incomplete_endings = [
        r'\bdan$', r'\batau$', r'\byang$', r'\bdi$', r'\bke$',
        r'\bdari$', r'\buntuk$', r'\bdengan$', r'\bpada$',
        r'\bini$', r'\bitu$', r'\badalah$',
        r'\bvia$', r'\bmelalui$', r'\blewat$',  # incomplete reference
        r'\bsilakan$', r'\bklik$', r'\bakses$',  # incomplete instructions
        r'\bseperti$', r'\byaitu$', r'\byakni$',  # incomplete explanations
        r'\bberkata$', r'\bmenyatakan$', r'\bmengatakan$',  # incomplete quotes
    ]
    for pattern in incomplete_endings:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    # Ends with colon followed by just punctuation (e.g., ": ." or ", : .")
    if re.search(r'[,:]\s*[.:!?]?\s*$', text):
        return True
    
    # Ends with just punctuation after comma (e.g., ", .")
    if re.search(r',\s*\.$', text):
        return True
    
    # Ends with incomplete abbreviation (e.g., "terj.", "dll.", "dkk.")
    # These are usually cut-off translations or incomplete references
    incomplete_abbrevs = [r'terj\.$', r'dll\.$', r'dkk\.$', r'dst\.$', r'etc\.$']
    for pattern in incomplete_abbrevs:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    # Ends with incomplete citation (e.g., "(Bil.", "(Mat.", "(Yoh.")
    if re.search(r'\([A-Z][a-z]{1,3}\.?$', text):
        return True
    
    # Ends with incomplete name (e.g., "Anna Ch.", "John D.", "Maria S.")
    if re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]?\.$', text):
        return True
    
    # Ends with opening parenthesis (incomplete citation)
    if text.endswith('('):
        return True
    
    return False


def is_boilerplate_start(text: str) -> bool:
    """
    Check if a sentence starts with boilerplate navigation patterns.
    
    Examples:
    - "Artikel / Video terkini..."
    - "Baca juga: ..."
    - "Tags: ..."
    """
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    for pattern in BOILERPLATE_START_PATTERNS:
        if re.match(pattern, text_lower, re.IGNORECASE):
            return True
    
    return False


def has_weird_start(text: str) -> bool:
    """
    Detect sentences with weird/incomplete starts.
    
    Examples:
    - ", 13-14 Desember" - starts with comma
    - "1. Pengantar" - starts with numbered list
    """
    if not text:
        return False
    
    text = text.strip()
    
    # Starts with comma or period
    if text[0] in ',.;:':
        return True
    
    # Starts with number followed by period (list item): "1. Something"
    if re.match(r'^\d{1,2}\.\s', text):
        return True
    
    # Starts with ellipsis
    if text.startswith('...') or text.startswith('…'):
        return True
    
    # Starts with metadata tags like "ORG-", "IMG-", "URL-", etc.
    if re.match(r'^[A-Z]{2,5}\s*[-:]\s*', text):
        return True
    
    # Starts with location/metadata prefixes
    # Patterns: "Com -", "Jakarta-", "Abepura, Papua -"
    if re.match(r'^(Com\s*[-–—]|Jakarta[-–—]|Papua[-–—]|[A-Z][a-z]+,\s*[A-Z][a-z]+\s*[-–—])', text):
        return True
    
    # Starts with author/date metadata
    # Pattern: "Pdt Name | 13-01-2024" or "Name, Title | Date"
    if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s*,\s*[A-Z]\.?\s*[A-Z]?\.?)?\s*[|]\s*\d', text):
        return True
    
    # Starts with academic citation like "(Ibid", "((Ibid", "(Op.cit"
    if re.match(r'^\(?[\(\[]\s*(?:Ibid|Op\.?\s*cit|Loc\.?\s*cit|cf\.?|p\.|pp\.)', text, re.IGNORECASE):
        return True
    
    # Starts with double parentheses (usually broken citation)
    if text.startswith('((') or text.startswith('[['):
        return True
    
    # Starts with lowercase after cleaning (likely fragment)
    if text[0].islower():
        return True
    
    return False


def is_schedule_or_time_fragment(text: str) -> bool:
    """
    Detect schedule/time fragments that aren't proper sentences.
    
    Examples:
    - "Misa Harian: 06.00 (kecuali Senin)"
    - "Sabtu: 18.00 WIB"
    - "Minggu: 06.30 WIB"
    """
    if not text:
        return False
    
    # Time schedule pattern: "Day: HH:MM" or "Event: HH.MM"
    if re.search(r':\s*\d{1,2}[.:]\d{2}\s*(WIB|WITA|WIT)?', text):
        # If sentence is mostly about time/schedule
        time_words = len(re.findall(r'\d{1,2}[.:]\d{2}', text))
        total_words = len(text.split())
        if time_words >= 1 and total_words < 15:
            return True
    
    return False


def is_just_names_or_attribution(text: str) -> bool:
    """
    Detect sentences that are just names/attributions.
    
    Examples:
    - "R. Danieli"
    - "Antonio Carigi (guardian)"
    - "Postinus Gulö"
    - "Vera boru Pangaribuan pada malam itu." (incomplete name fragment)
    """
    if not text:
        return False
    
    words = text.split()
    
    # Very short and looks like a name
    if len(words) <= 4:
        # Check if mostly capitalized words (names)
        cap_words = sum(1 for w in words if w[0].isupper() if w)
        if cap_words >= len(words) * 0.75:
            # Check for name patterns
            if re.match(r'^[A-Z][a-z]*\.?\s+[A-Z]', text):
                return True
            # Check for (role) suffix
            if re.search(r'\([a-z]+\)\s*$', text):
                return True
    
    # Check for incomplete name fragments (name + very short phrase)
    # Pattern: "Vera boru Pangaribuan pada malam itu."
    if len(words) <= 6:
        # Starts with name-like pattern (2-3 capitalized words)
        if re.match(r'^[A-Z][a-z]+\s+[a-z]+\s+[A-Z][a-z]+', text):
            # Ends with common Indonesian words that suggest incompleteness
            incomplete_endings = ['pada', 'di', 'dengan', 'untuk', 'dari', 'ke', 'itu', 'ini']
            if words[-1].lower() in incomplete_endings or words[-1].endswith('.'):
                return True
    
    return False


def fix_mojibake(text: str) -> str:
    """Fix encoding issues (mojibake)."""
    if not text:
        return ""
    
    # Use ftfy if available (much more comprehensive)
    if FTFY_AVAILABLE:
        text = ftfy.fix_text(text)
    
    # Apply manual replacements as fallback/supplement
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    
    return text


def protect_titles(text: str) -> str:
    """Remove dots from titles to prevent bad sentence splitting."""
    for pattern, replacement in TITLE_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


def strip_nav_keywords(text: str) -> str:
    """Remove navigation/sidebar keywords."""
    for keyword in NAV_KEYWORDS:
        # Case-insensitive word boundary removal
        pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
        text = pattern.sub('', text)
    return text


def remove_references(text: str) -> str:
    """Remove Bible/Quran references and metadata."""
    text = BIBLE_REF_PATTERN.sub('', text)
    text = QURAN_REF_PATTERN.sub('', text)
    
    for pattern in METADATA_PATTERNS:
        text = pattern.sub('', text)
    
    return text


def remove_arabic_script(text: str) -> str:
    """Remove Arabic script characters while preserving surrounding text."""
    return ARABIC_SCRIPT_PATTERN.sub(' ', text)


def normalize_text(text: str) -> str:
    """Normalize quotes, remove non-printables, collapse whitespace."""
    if not text:
        return ""
    
    # Convert smart quotes to straight quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('–', '-').replace('—', '-')
    
    # Remove non-printable unicode characters (Category "C")
    text = ''.join(char for char in text if not unicodedata.category(char).startswith('C'))
    
    # Collapse multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Collapse multiple newlines to double newline
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    
    return text.strip()


def clean_text_pipeline(text: str) -> str:
    """
    Apply the complete text cleaning pipeline in order:
    1. Mojibake fixes
    2. Drop Cap / Missing letter fixes
    3. Title protection
    4. Nav/Sidebar stripping
    5. Reference removal
    6. Arabic script removal
    7. Normalization
    """
    if not text:
        return ""
    
    text = fix_mojibake(text)
    text = fix_common_typos(text)  # Fix "Drop Cap" missing letters
    text = protect_titles(text)
    text = strip_nav_keywords(text)
    text = remove_references(text)
    text = remove_arabic_script(text)
    text = normalize_text(text)
    
    return text


# =============================================================================
# 4. STRUCTURAL NOISE FILTERING (SENTENCE LEVEL)
# =============================================================================

# Sentence length limits for classifier quality
MIN_SENTENCE_CHARS = 25      # Minimum characters (was 12)
MAX_SENTENCE_CHARS = 500     # Maximum characters (prevent concatenated junk)
MIN_SENTENCE_WORDS = 5       # Minimum words (was 2)
MAX_SENTENCE_WORDS = 100     # Maximum words

# Whitelist phrases to always keep (religious expressions)
WHITELIST_PHRASES = {
    'amin', 'alleluia', 'hallelujah', 'puji tuhan', 'selamat paskah',
    'selamat natal', 'selamat idul fitri', 'selamat lebaran',
    'alhamdulillah', 'subhanallah', 'masya allah', 'insya allah',
    'assalamualaikum', 'bismillah', 'astaghfirullah', 'syukur',
}

# Boilerplate phrases to REMOVE (navigation, UI elements)
BOILERPLATE_PHRASES = [
    # Navigation
    'skip to content', 'lihat lebih banyak', 'baca selengkapnya', 'read more',
    'lihat semua', 'view all', 'see more', 'load more', 'show more',
    'baca juga', 'artikel terkait', 'berita terkait', 'related posts',
    'older posts', 'newer posts', 'next page', 'previous page',
    'halaman selanjutnya', 'halaman sebelumnya', 'category archives',
    # Social/Sharing
    'share this', 'bagikan ini', 'ikuti kami', 'follow us',
    'subscribe', 'berlangganan', 'join our', 'gabung dengan',
    # Login/Forms
    'log into your account', 'your username', 'your password',
    'forgot your password', 'reset your password', 'sign in', 'sign up',
    'create account', 'register here', 'daftar sekarang',
    'enter your email', 'type your email',
    # Comments
    'no comments yet', 'be the first', 'leave a comment', 'add a comment',
    'comments (0)', 'komentar (0)',
    # Cookie/Privacy
    'cookie policy', 'privacy policy', 'terms of service',
    'penggunaan cookie', 'setuju atau batalkan', 'cookies dalam situs',
    'kami menggunakan cookie', 'mengakses kami menggunakan cookie',
    'kenyamanan anda selama mengakses', 'cookies untuk memastikan',
    # Media errors
    'you need javascript', 'javascript enabled', 'enable javascript',
    'unable to retrieve', 'error loading', 'no videos found',
    'api key', 'invalid channel',
    # Gambling/Spam
    'slot gacor', 'slot online', 'jackpot', 'rtp slot', 'indobet',
    'situs slot', 'main slot', 'judi online', 'poker online',
    'togel online', 'live casino', 'deposit pulsa', 'bonus deposit',
    'rtp real-time', 'peluang menang', 'provider game', 'situs terpercaya',
    'game dengan rtp', 'makin tinggi rtp', 'update info rtp',
    # Payment/Commercial
    'biaya pendaftaran ditransfer', 'klik/tap pada gambar untuk memesan',
    'transfer ke rekening', 'nomor rekening', 'biaya registrasi',
    # Misc UI
    'click here', 'klik di sini', 'tap here', 'press here',
    'ingin berlangganan', 'silahkan klik', 'akses kalkulator',
    'required fields', 'fields are marked',
    'save my name', 'time i comment',
    'cancel reply', 'post comment',
    # Attribution/Credits
    'created by', 'designed by', 'powered by', 'developed by',
    # Calls to action
    'dapatkan update', 'mohon doa', 'di sini!',
    # Schedule fragments
    'misa harian:', 'jadwal misa', 'jadwal ibadah',
]

# Short connector words that shouldn't be standalone sentences
# These are incomplete fragments when alone
INCOMPLETE_STARTERS = {
    'namun', 'lalu', 'bahkan', 'kemudian', 'selanjutnya', 'pertama',
    'kedua', 'ketiga', 'keempat', 'dengan demikian', 'oleh karena itu',
    'oleh sebab itu', 'karena itu', 'sebab itu', 'maka', 'jadi',
    'sehingga', 'padahal', 'sedangkan', 'sebaliknya', 'begitu pula',
    'demikian pula', 'apa itu', 'berikut ini', 'di antaranya',
}

# Metadata signature patterns
METADATA_SIGNATURE_PATTERN = re.compile(r'\((?:Kontributor|Editor|Penulis|Fotografer|Sumber):\s*[^)]+\)', re.IGNORECASE)

# URL pattern to strip from sentences
URL_PATTERN = re.compile(r'https?://[^\s]+|www\.[^\s]+|\w+\.com[^\s]*|\w+\.org[^\s]*|\w+\.id[^\s]*|\w+\.net[^\s]*', re.IGNORECASE)

# High nav-density keywords
NAV_DENSITY_KEYWORDS = {
    'home', 'search', 'login', 'view', 'tags', 'menu', 'page', 'next',
    'previous', 'click', 'back', 'top', 'share', 'print', 'more',
    'read', 'see', 'all', 'show', 'hide', 'close', 'open', 'skip',
    'subscribe', 'newsletter', 'follow', 'join', 'register', 'sign',
}


def clean_sentence(sentence: str) -> str:
    """
    Clean a sentence by removing URLs, citations, fixing punctuation, and normalizing.
    """
    if not sentence:
        return ""
    
    # Remove URLs
    sentence = URL_PATTERN.sub('', sentence)
    
    # Remove Bible/Quran citations at the start of sentences
    # Patterns: (Mat 25:46), (Yoh 3:16), (1 Kor 13:4-7), (QS Al-Baqarah: 183)
    # Also catches incomplete citations: (/90: 103), (minun/23: 4), (/30: 39)
    sentence = re.sub(
        r'^\s*\([A-Za-z0-9\s\.\-:/]+\)\s*',
        '',
        sentence
    )
    
    # Remove incomplete citations that start with / (e.g., "/90: 103" or "minun/23: 4")
    sentence = re.sub(
        r'^\s*\(/?[A-Za-z]*/?\s*\d+[:\s]+\d+[^)]*\)\s*',
        '',
        sentence,
        flags=re.IGNORECASE
    )
    
    # Remove inline verse references: (ayat 47), (Bil. 6:1-21), (Kel. 13)
    sentence = re.sub(r'\(ayat\s*\d+[^)]*\)', '', sentence, flags=re.IGNORECASE)
    sentence = re.sub(r'\([A-Za-z]{2,4}\.\s*\d+[^)]*\)', '', sentence)
    
    # Remove any remaining parenthetical citations with numbers (Quran/Bible refs)
    # Pattern: (anything with numbers and colons)
    sentence = re.sub(r'\([^)]*\d+[:\s]+\d+[^)]*\)', '', sentence)
    
    # Remove parenthetical titles/subtitles (long parentheticals that look like titles)
    # Pattern: (Refleksi Syukur dan Doa Ulang Tahun Kelahiran) - subtitles that got merged
    # Remove if it's 20+ chars, mostly title case words, and no numbers/colons
    long_parentheticals = re.finditer(r'\(([^)]{20,})\)', sentence)
    for match in list(long_parentheticals):  # Convert to list to avoid modification during iteration
        content = match.group(1)
        words = content.split()
        # If 4+ words, mostly title case, and no numbers/colons, it's likely a title
        if len(words) >= 4:
            title_case_count = sum(1 for w in words if w and w[0].isupper())
            if title_case_count >= len(words) * 0.6 and not re.search(r'[\d:]', content):
                sentence = sentence.replace(match.group(0), ' ', 1)  # Replace first occurrence
    
    # Remove location/metadata prefixes at start
    # Patterns: "Com - Jayapura, 04 Juni 2025-", "Abepura, Papua -", "Jakarta- -", "Com Nabire, Papua -"
    # First remove "Com - " or "Com " prefix
    sentence = re.sub(r'^Com\s*[-–—]?\s*', '', sentence)
    # Then remove location + date patterns: "Jayapura, 04 Juni 2025-", "Nabire, Papua -"
    sentence = re.sub(r'^[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+)*(?:\s*,\s*\d{1,2}\s+[A-Z][a-z]+\s+\d{4})?\s*[-–—]\s*', '', sentence)
    # Remove double dashes/spaces
    sentence = re.sub(r'[-–—]\s*[-–—]', ' ', sentence)
    sentence = re.sub(r'\s+[-–—]\s+', ' ', sentence)
    
    # Remove author/date metadata at start
    # Pattern: "Pdt Imanuel Ginting, S. Th | 13-01-2024" or "Name | Date"
    # Match author name (at least 2 words), anything up to pipe, then date
    sentence = re.sub(r'^[A-Z][a-z]+\s+[A-Z][a-z]+.*?[|]\s*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', '', sentence)
    
    # Remove [...] placeholders (truncated content markers)
    sentence = re.sub(r'\[…\]', '', sentence)
    sentence = re.sub(r'\[\.\.\.\]', '', sentence)
    
    # Remove "Created by" and similar attribution phrases
    sentence = re.sub(
        r'Created by\s+\w+.*$|Designed by\s+\w+.*$|Powered by\s+\w+.*$',
        '',
        sentence,
        flags=re.IGNORECASE
    )
    
    # Remove bank account patterns: BSI: 7086882242a.n.
    sentence = re.sub(r'\b[A-Z]{2,5}:\s*\d{8,}[a-z\.]*', '', sentence)
    
    # Remove separator lines (===, ---, ***)
    sentence = re.sub(r'[=\-\*]{3,}', ' ', sentence)
    
    # Remove donation/action banners that got concatenated
    sentence = re.sub(r'Para dermawan bisa donasi.*$', '', sentence, flags=re.IGNORECASE)
    sentence = re.sub(r'dengan mengklik banner.*$', '', sentence, flags=re.IGNORECASE)
    
    # Remove "Jumlah Pembaca: 134" patterns (view count metadata) - keep text after
    sentence = re.sub(r'^Jumlah\s+Pembaca\s*:\s*\d+\s+', '', sentence, flags=re.IGNORECASE)
    
    # Remove "Baca online: ..." navigation patterns - remove entire pattern
    sentence = re.sub(r'Baca\s+online\s*:.*$', '', sentence, flags=re.IGNORECASE)
    sentence = re.sub(r'Pelajaran\s+\d+\s*\|\s*Pertanyaan\s+\d+\s*\|\s*Referensi\s+\d+.*$', '', sentence, flags=re.IGNORECASE)
    
    # Remove name lists with "Wil X : Pdt Name" patterns
    # Pattern: "Name- Wil 8 : Pdt Name- Wil 9 : Pdt Name..."
    sentence = re.sub(r'[A-Z][a-z]+\s+[A-Z][a-z]+-\s*Wil\s+\d+\s*:\s*Pdt\s+[A-Z][a-z]+.*$', '', sentence)
    
    # Remove copyright symbols and surrounding text
    sentence = re.sub(r'©.*$', '', sentence)
    
    # Fix punctuation stuck to next word (e.g., "?RENUNGAN" → "? RENUNGAN")
    sentence = re.sub(r'([.!?])([A-Z])', r'\1 \2', sentence)
    
    # Fix closing parenthesis stuck to next word
    sentence = re.sub(r'\)([A-Z])', r') \1', sentence)
    
    # Fix colon stuck to next word (e.g., "Rasul:Artinya" → "Rasul: Artinya")
    # But don't break time/verse references like "3:16" or "10:30"
    sentence = re.sub(r':([A-Za-z])', r': \1', sentence)
    
    # Remove double commas and weird punctuation patterns
    sentence = re.sub(r',\s*,', ',', sentence)
    sentence = re.sub(r'\.\s*\.', '.', sentence)
    sentence = re.sub(r'\.{3,}', '...', sentence)  # Normalize ellipsis
    
    # Remove numbered list markers stuck to text: "zakat:1." or "#2:"
    sentence = re.sub(r':\d+\.?\s*$', '', sentence)
    sentence = re.sub(r'#\d+[:\.]', '', sentence)
    
    # Strip ALL types of leading/trailing quotation marks (including smart quotes)
    # Using explicit Unicode code points to ensure we catch all quote types:
    # U+0022 " straight double quote
    # U+0027 ' straight single quote  
    # U+201C " left double quotation mark
    # U+201D " right double quotation mark
    # U+2018 ' left single quotation mark
    # U+2019 ' right single quotation mark
    # U+201E „ double low-9 quotation mark
    # U+201F ‟ double high-reversed-9 quotation mark
    # U+00AB « left-pointing double angle quotation mark
    # U+00BB » right-pointing double angle quotation mark
    quote_chars = (
        '"\'`'                    # ASCII quotes
        '\u201c\u201d'            # " " curly double quotes
        '\u2018\u2019'            # ' ' curly single quotes
        '\u201e\u201f'            # „ ‟ other double quotes
        '\u00ab\u00bb'            # « » guillemets
        '\u3008\u3009'            # 〈 〉 angle brackets
        '\u300a\u300b'            # 《 》 double angle brackets
        '\u300c\u300d'            # 「 」 corner brackets
        '\u300e\u300f'            # 『 』 white corner brackets
        '\u2039\u203a'            # ‹ › single guillemets
    )
    sentence = sentence.strip()
    # Keep stripping quotes until none left at edges
    max_iterations = 10  # Safety limit
    for _ in range(max_iterations):
        if not sentence:
            break
        changed = False
        if sentence[0] in quote_chars:
            sentence = sentence[1:].strip()
            changed = True
        if sentence and sentence[-1] in quote_chars:
            sentence = sentence[:-1].strip()
            changed = True
        if not changed:
            break
    
    # Strip leading/trailing ellipsis
    sentence = re.sub(r'^\.{2,}\s*', '', sentence)
    sentence = re.sub(r'\s*\.{2,}$', '', sentence)
    
    # Remove internal quotes (quotes that are not at edges)
    # This cleans up quoted speech/citations for cleaner training data
    # e.g., 'Dia berkata, "halo"' → 'Dia berkata, halo'
    sentence = sentence.replace('"', '')
    sentence = sentence.replace("'", '')
    
    # Normalize whitespace
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    
    return sentence


def is_incomplete_fragment(text: str) -> bool:
    """
    Check if text is just an incomplete connector/fragment.
    
    Examples: "Namun", "Oleh karena itu", "Pertama", "Tapi", "Sebab"
    """
    if not text:
        return True
    
    text_lower = text.lower().strip()
    words = text_lower.split()
    
    # Very short sentences (1-2 words) are usually fragments
    if len(words) <= 2:
        return True
    
    # Check if it's just an incomplete starter word/phrase
    if text_lower in INCOMPLETE_STARTERS:
        return True
    
    # Check if it starts with incomplete starter and is very short (< 8 words)
    for starter in INCOMPLETE_STARTERS:
        if text_lower.startswith(starter) and len(words) < 8:
            return True
    
    # Short sentences starting with common connectors
    short_connectors = {'tapi', 'sebab', 'kalo', 'kalau', 'jika', 'bila', 'saat', 'ketika', 
                        'aku', 'kita', 'kami', 'gimana', 'bagaimana', 'apakah', 'maka'}
    if len(words) < 8 and words[0] in short_connectors:
        return True
    
    # Questions that are incomplete (no question mark or too short)
    question_starters = {'bagaimana', 'gimana', 'apakah', 'kenapa', 'mengapa', 'siapa', 
                         'dimana', 'kapan', 'berapa', 'apa'}
    if words[0] in question_starters and len(words) < 10 and '?' not in text:
        return True
    
    return False


def contains_boilerplate(text: str) -> bool:
    """Check if text contains boilerplate navigation/UI phrases."""
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in BOILERPLATE_PHRASES)


def contains_whitelist_phrase(text: str) -> bool:
    """
    Check if text contains a whitelisted religious phrase.
    Uses word boundaries to avoid false positives (e.g., "streaming" contains "amin").
    """
    text_lower = text.lower()
    for phrase in WHITELIST_PHRASES:
        # Use word boundary matching to avoid false positives
        pattern = r'\b' + re.escape(phrase) + r'\b'
        if re.search(pattern, text_lower):
            return True
    return False


def has_metadata_signature(text: str) -> bool:
    """Check if sentence contains metadata signatures."""
    return bool(METADATA_SIGNATURE_PATTERN.search(text))


def calculate_nav_density(text: str) -> float:
    """Calculate percentage of navigation keywords in text."""
    words = text.lower().split()
    if not words:
        return 0.0
    nav_count = sum(1 for word in words if word in NAV_DENSITY_KEYWORDS)
    return nav_count / len(words)


def calculate_caps_density(text: str) -> float:
    """Calculate percentage of ALL CAPS words."""
    words = text.split()
    if not words:
        return 0.0
    caps_count = sum(1 for word in words if word.isupper() and len(word) > 1)
    return caps_count / len(words)


def has_proper_start(text: str) -> bool:
    """Check if sentence starts with uppercase, quote, or parenthesis."""
    if not text:
        return False
    first_char = text[0]
    return first_char.isupper() or first_char in '"\'("'


def looks_like_name_list(text: str) -> bool:
    """
    Detect if text is a name list (directory/contact list).
    
    Examples:
    - "Nimrod Sesa- Wil 8 : Pdt Lewi Sawor- Wil 9 : Pdt Petrus Womsiwor..."
    - "Name- Section X : Title Name- Section Y : Title Name..."
    """
    # Pattern: Name- Section X : Title Name- Section Y : Title...
    if re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+-\s*(?:Wil|Section|Bagian)\s+\d+\s*:\s*(?:Pdt|Dr|Mr|Mrs)', text):
        return True
    
    # Pattern: Multiple "Name- Section : Title" sequences
    name_list_patterns = len(re.findall(r'[A-Z][a-z]+\s+[A-Z][a-z]+-\s*\w+\s+\d+\s*:', text))
    if name_list_patterns >= 2:
        return True
    
    return False


def looks_like_concatenated_menu(text: str) -> bool:
    """
    Detect if text looks like concatenated menu items or navigation.
    
    Pattern: Multiple capitalized words without proper sentence structure.
    Example: "BeritaArtikelVideoGaleriKontak" or "Home Menu Tentang Kontak"
    """
    # Check for CamelCase concatenation (no spaces between capitalized words)
    if re.search(r'[a-z][A-Z]', text):
        # Count transitions - too many suggests concatenated items
        transitions = len(re.findall(r'[a-z][A-Z]', text))
        if transitions >= 3:
            return True
    
    # Check for menu-like pattern (mostly short capitalized words)
    words = text.split()
    if len(words) >= 4:
        short_caps = sum(1 for w in words if len(w) <= 12 and w[0].isupper() if w)
        if short_caps / len(words) > 0.7:
            # Likely a menu list
            return True
    
    return False


def is_valid_sentence(sentence: str, already_cleaned: bool = False) -> Tuple[bool, str]:
    """
    Validate a sentence against structural noise filters.
    
    This is designed to keep secular content from religious sources
    while filtering out navigation, boilerplate, and junk.
    
    Args:
        sentence: The sentence to validate
        already_cleaned: If True, skip redundant cleaning (optimization)
    
    Returns:
        Tuple of (is_valid: bool, rejection_reason: str)
    """
    # Skip cleaning if already done by caller
    if not already_cleaned:
        sentence = clean_sentence(sentence)
    
    if not sentence:
        return False, "empty_after_clean"
    
    # =========== LENGTH CHECKS ===========
    
    # Too short (not enough content)
    if len(sentence) < MIN_SENTENCE_CHARS:
        return False, "too_short_chars"
    
    # Too long (likely concatenated junk)
    if len(sentence) > MAX_SENTENCE_CHARS:
        return False, "too_long_chars"
    
    words = sentence.split()
    
    # Too few words (fragments)
    if len(words) < MIN_SENTENCE_WORDS:
        return False, "too_few_words"
    
    # Too many words (paragraph dumps)
    if len(words) > MAX_SENTENCE_WORDS:
        return False, "too_many_words"
    
    # =========== WHITELIST CHECK ===========
    # Only for SHORT religious phrases (bypasses length filters, not noise filters)
    is_whitelisted = contains_whitelist_phrase(sentence)
    if is_whitelisted and len(words) <= 5:
        # Short religious phrases like "Amin" or "Puji Tuhan" get a pass
        return True, "whitelisted"
    
    # =========== NOISE FILTERS ===========
    # These run even for whitelisted phrases (to catch bad concatenations etc.)
    
    # Incomplete fragments (just connector words)
    if is_incomplete_fragment(sentence):
        return False, "incomplete_fragment"
    
    # Boilerplate navigation/UI phrases
    if contains_boilerplate(sentence):
        return False, "boilerplate"
    
    # Address/Phone junk
    if is_address_junk(sentence):
        return False, "address_junk"
    
    # Bad concatenation (dates/words stuck together)
    if has_bad_concatenation(sentence):
        return False, "bad_concatenation"
    
    # Weird/incomplete endings
    if has_weird_ending(sentence):
        return False, "weird_ending"
    
    # Weird/incomplete starts
    if has_weird_start(sentence):
        return False, "weird_start"
    
    # Boilerplate navigation starts
    if is_boilerplate_start(sentence):
        return False, "boilerplate_start"
    
    # Concatenated menu items
    if looks_like_concatenated_menu(sentence):
        return False, "concatenated_menu"
    
    # Name lists (directory/contact lists)
    if looks_like_name_list(sentence):
        return False, "name_list"
    
    # Metadata signatures
    if has_metadata_signature(sentence):
        return False, "metadata_signature"
    
    # Schedule/time fragments
    if is_schedule_or_time_fragment(sentence):
        return False, "schedule_fragment"
    
    # Just names or attributions
    if is_just_names_or_attribution(sentence):
        return False, "names_only"
    
    # Nav density (>30% is noise - lowered from 35%)
    if calculate_nav_density(sentence) > 0.30:
        return False, "high_nav_density"
    
    # Caps density (>60% with >6 words is noise - made stricter)
    if len(words) > 6 and calculate_caps_density(sentence) > 0.60:
        return False, "high_caps_density"
    
    # Proper start check (uppercase, quote, or parenthesis)
    if not has_proper_start(sentence):
        return False, "improper_start"
    
    # =========== PASSED ALL FILTERS ===========
    return True, "valid"


# =============================================================================
# 5. SYSTEMATIC DROP CAP HTML FIX (Pre-processing)
# =============================================================================

def fix_drop_caps_html(html: str) -> str:
    """
    SYSTEMATIC fix for drop cap styling in HTML BEFORE text extraction.
    
    The Problem:
        Websites style the first letter(s) of paragraphs in separate HTML elements.
        When trafilatura extracts text, it often strips these styled elements,
        leaving broken words like "ati" instead of "Hati".
    
    Examples of drop cap HTML patterns:
        <span class="dropcap">H</span>ati
        <span style="float:left; font-size:3em">Ma</span>sjid
        <div class="first-letter">H</div>omili
        <span class="initial">B</span>erkomunikasi
    
    Solution:
        We use BeautifulSoup to find ALL inline elements (span, div, etc.) that:
        1. Contain only 1-3 uppercase/titlecase letters
        2. Are immediately followed by lowercase text
        
        Then we merge the letter(s) back into the text, removing the element.
    """
    if not html:
        return html
    
    try:
        from bs4 import BeautifulSoup, NavigableString
        
        soup = BeautifulSoup(html, 'lxml')
        
        # Find all potential drop cap elements
        # These are typically spans/divs with very short text content
        for tag in soup.find_all(['span', 'div', 'p', 'b', 'strong', 'i', 'em']):
            # Get the text content of this tag
            tag_text = tag.get_text(strip=True)
            
            # Skip if empty or too long (drop caps are 1-3 chars)
            if not tag_text or len(tag_text) > 3:
                continue
            
            # Skip if not letters only
            if not tag_text.isalpha():
                continue
            
            # Skip if all lowercase (drop caps are usually uppercase/titlecase)
            if tag_text.islower():
                continue
            
            # Check if this element has drop cap indicators in class or style
            tag_class = ' '.join(tag.get('class', []))
            tag_style = tag.get('style', '')
            
            is_likely_dropcap = False
            
            # Method 1: Check class name for drop cap keywords
            dropcap_keywords = ['drop', 'initial', 'first', 'cap', 'big', 'letter', 'lead']
            if any(kw in tag_class.lower() for kw in dropcap_keywords):
                is_likely_dropcap = True
            
            # Method 2: Check inline style for drop cap patterns
            dropcap_styles = ['float', 'font-size:', 'line-height:', 'display:block']
            if any(s in tag_style.lower() for s in dropcap_styles):
                is_likely_dropcap = True
            
            # Method 3: Single uppercase letter followed by lowercase sibling
            # This is the most universal detection
            if len(tag_text) <= 2 and tag_text[0].isupper():
                next_sibling = tag.next_sibling
                if next_sibling:
                    # Get the text immediately after this element
                    if isinstance(next_sibling, NavigableString):
                        next_text = str(next_sibling).lstrip()
                        # If next text starts with lowercase, likely a drop cap
                        if next_text and next_text[0].islower():
                            is_likely_dropcap = True
            
            # If we detected a drop cap, merge it with the following text
            if is_likely_dropcap:
                next_sibling = tag.next_sibling
                if next_sibling and isinstance(next_sibling, NavigableString):
                    # Merge: "H" + "ati" → "Hati"
                    merged_text = tag_text + str(next_sibling)
                    # Replace the element and sibling with merged text
                    next_sibling.replace_with('')
                    tag.replace_with(merged_text)
        
        return str(soup)
        
    except Exception:
        # If BeautifulSoup fails, fall back to regex patterns
        return _fix_drop_caps_regex(html)


def _fix_drop_caps_regex(html: str) -> str:
    """
    Regex fallback for drop cap fixing when BeautifulSoup isn't available.
    Less comprehensive but still catches common patterns.
    """
    if not html:
        return html
    
    # Pattern 1: Class-based drop caps
    # <span class="...drop...">X</span>rest → Xrest
    pattern1 = re.compile(
        r'<(?:span|div)[^>]*class="[^"]*(?:drop|initial|first|big|cap|letter)[^"]*"[^>]*>'
        r'([A-Za-z]{1,3})'
        r'</(?:span|div)>\s*'
        r'([a-z]+)',
        re.IGNORECASE
    )
    html = pattern1.sub(r'\1\2', html)
    
    # Pattern 2: Style-based drop caps (float, large font)
    # <span style="float:left">X</span>rest → Xrest
    pattern2 = re.compile(
        r'<(?:span|div)[^>]*style="[^"]*(?:float|font-size:\s*[2-9])[^"]*"[^>]*>'
        r'([A-Za-z]{1,3})'
        r'</(?:span|div)>\s*'
        r'([a-z]+)',
        re.IGNORECASE
    )
    html = pattern2.sub(r'\1\2', html)
    
    # Pattern 3: Generic single uppercase letter in span followed by lowercase
    # <span>H</span>ati → Hati (most universal)
    pattern3 = re.compile(
        r'<(?:span|div|b|strong)[^>]*>'
        r'([A-Z]{1,2})'
        r'</(?:span|div|b|strong)>\s*'
        r'([a-z]{2,})',
        re.IGNORECASE
    )
    html = pattern3.sub(r'\1\2', html)
    
    return html


# =============================================================================
# 6. MAIN EXTRACTION FUNCTION
# =============================================================================

def extract_sentences(html_content: str) -> List[dict]:
    """
    Extract and clean sentences from HTML content.
    
    Returns:
        List of dicts with keys: 'sentence', 'detected_lang', 'status'
    """
    results = []
    
    # CRITICAL: Fix drop caps in HTML BEFORE extraction
    html_content = fix_drop_caps_html(html_content)
    
    # Try trafilatura for main content extraction (better than BeautifulSoup)
    try:
        import trafilatura
        text = trafilatura.extract(html_content, include_comments=False, include_tables=False)
        if not text:
            # Fallback to basic extraction
            text = _fallback_extract(html_content)
    except Exception:
        text = _fallback_extract(html_content)
    
    if not text:
        return results
    
    # Apply cleaning pipeline
    text = clean_text_pipeline(text)
    
    if not text:
        return results
    
    # Pre-process for better sentence splitting
    # Convert double dots to proper sentence boundary
    text = re.sub(r'\.\.+\s*', '. ', text)
    # Fix missing space after punctuation
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    
    # Split into sentences
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        sentences = re.split(r'[.!?]+\s+', text)
    
    # Process each sentence
    for sentence in sentences:
        sentence = sentence.strip()
        
        if not sentence:
            continue
        
        # Clean the sentence first (remove URLs, normalize)
        cleaned = clean_sentence(sentence)
        
        if not cleaned:
            continue
        
        # Validate structure (skip redundant cleaning since we just did it)
        is_valid, validity_reason = is_valid_sentence(cleaned, already_cleaned=True)
        if not is_valid:
            continue
        
        # Language detection on cleaned sentence
        is_indonesian, detected_lang = is_indonesian_robust(cleaned)
        
        results.append({
            'sentence': cleaned,  # Return the cleaned version
            'detected_lang': detected_lang,
            'is_indonesian': is_indonesian,
        })
    
    return results


def _fallback_extract(html_content: str) -> str:
    """Fallback HTML extraction using BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove unwanted elements
        for tag in ['script', 'style', 'nav', 'header', 'footer', 'aside',
                    'form', 'button', 'input', 'select', 'noscript', 'iframe']:
            for element in soup.find_all(tag):
                element.decompose()
        
        return soup.get_text(separator=' ')
    except Exception:
        return ""


def extract_sentences_simple(html_content: str) -> List[str]:
    """
    Simplified extraction that returns only valid Indonesian sentences.
    Used for backward compatibility with existing crawler.
    """
    results = extract_sentences(html_content)
    return [r['sentence'] for r in results if r['is_indonesian']]
