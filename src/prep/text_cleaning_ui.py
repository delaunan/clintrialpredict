import re
from typing import Optional

# ------------------------------------------------------------------------------
# UI FORMATTING HELPERS
# ------------------------------------------------------------------------------

def ui_truncate(text: str, limit: int) -> str:
    """Truncates text to a limit and adds ellipsis if exceeded."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit - 3].strip() + "..."

def ui_smart_title_case(text: str) -> str:
    """For TITLES: Converts ALL CAPS to Title Case; leaves mixed-case alone."""
    if not text:
        return ""
    if text.isupper():
        return text.title()
    return text

def ui_smart_sentence_case(text: str) -> str:
    """For PHRASES: Converts ALL CAPS to Sentence Case; leaves mixed-case alone."""
    if not text:
        return ""
    if len(text) < 100 and text.isupper():
        return text.capitalize()
    return text

# ------------------------------------------------------------------------------
# CLEANUP PATTERNS (PRESERVE CONTENT, REMOVE SIGNS)
# ------------------------------------------------------------------------------

# Safety: Only match actual HTML tags (starts with < and a letter or /)
_HTML_TAGS = re.compile(r"(?is)<(?:[a-z/][^>]*|![^>]*|\[[^>]*\])>")

_CLINICAL_SYMBOLS = [
    # 1. HTML Entities (Survivors)
    (re.compile(r"&amp;"), "&"),
    (re.compile(r"&lt;"), "<"),
    (re.compile(r"&gt;"), ">"),
    (re.compile(r"&quot;"), '"'),
    (re.compile(r"&apos;"), "'"),
    (re.compile(r"&nbsp;"), " "),

    # 2. Math & Comparisons
    (re.compile(r"\\>="), "≥"),
    (re.compile(r"\\<="), "≤"),
    (re.compile(r"\\>"), ">"),
    (re.compile(r"\\<"), "<"),
    (re.compile(r"\\"), "<"), 
    (re.compile(r"\\+/-"), "±"),
    (re.compile(r"\\deg"), "°"),
    (re.compile(r"(?i)\bx\s*(\d+)"), r"× \1"), # 3x -> 3×
    (re.compile(r"(?i)(\d+)\s*x\b"), r"\1×"),  # 3x -> 3×

    # 3. Superscripts (Common in BSA / Lab values)
    (re.compile(r"(?i)\<?\^2"), "²"), 
    (re.compile(r"(?i)\<?\^3"), "³"),
    (re.compile(r"(?i)\<?\^4"), "⁴"),
    (re.compile(r"(?i)\<?\^9"), "⁹"),

    # 4. Greek / Clinical Units (Standardization)
    (re.compile(r"\b(u|mu)\s*g/"), "μg/"), # ug/mL -> μg/mL
    (re.compile(r"\b(u|mu)\s*L\b"), "μL"),  # uL -> μL
    (re.compile(r"\b(u|mu)\s*mol\b"), "μmol"),
    (re.compile(r"(?i)\b(alpha)\b"), "α"),
    (re.compile(r"(?i)\b(beta)\b"), "β"),
    (re.compile(r"(?i)\b(gamma)\b"), "γ"),
    (re.compile(r"(?i)\b(delta)\b"), "Δ"),

    # 5. Business / Legal
    (re.compile(r"\(R\)"), "®"),
    (re.compile(r"\(TM\)"), "™"),
    (re.compile(r"\(C\)"), "©"),

    # 6. Separators
    (re.compile(r"\\~"), "~"),
]

_WS = re.compile(r"[ \t]+")

# ------------------------------------------------------------------------------
# MAIN UI CLEANER
# ------------------------------------------------------------------------------

def ui_clean_text(text: Optional[str]) -> str:
    """Standard cleaner for single-line fields."""
    if not text or str(text).lower() == "nan": return ""
    t = str(text).strip()
    t = _HTML_TAGS.sub(" ", t)
    for pat, repl in _CLINICAL_SYMBOLS:
        t = pat.sub(repl, t)
    return _WS.sub(" ", t).strip()

def ui_format_multiline(text: Optional[str]) -> str:
    """
    Robust Multiline Formatter (v5.8).
    Comprehensive scientific symbol mapping and aggressive semantic splitting.
    """
    if not text or str(text).lower() == "nan": return ""
    
    # 1. Strip actual tags ONLY
    t = str(text).strip()
    # Normalize NBSP and other invisible control characters
    t = t.replace("\xa0", " ").replace("\t", " ")
    t = _HTML_TAGS.sub(" ", t)
    
    # 2. Forensic: Strip AACT Bracket Tags and Legacy Pointers
    t = t.replace("<[", "").replace("<]", "").replace("[<", "")
    t = t.replace("<- ", " ").replace("-> ", " ").replace("--> ", " ")
    
    # 3. Translate symbols AFTER stripping HTML to avoid range deletion
    for pat, repl in _CLINICAL_SYMBOLS:
        t = pat.sub(repl, t)

    # 4. Structural normalization
    # Force newline before major sections
    t = re.sub(r'([\.?;:~]|\s)\s*(Inclusion Criteria|Exclusion Criteria)', r'\1\n\2\n', t, flags=re.IGNORECASE)
    
    # Aggressive Semantic Splitting: Force newline before keywords if they are mid-line
    t = re.sub(r'\s+(NAME|DESC|TITLE|TIMEFRAME|MEASURE|ENDPOINT|OBJECTIVE|DOSING):', r'\n\1:', t, flags=re.IGNORECASE)
    
    t = t.replace("~", "\n").replace(" || ", "\n").replace("- -", "\n")
    
    processed = []
    last_header = ""
    # State tracking for spacing
    last_line_type = None 
    
    for line in t.splitlines():
        # A. Raw Bleaching: Remove ALL non-printable characters
        line = "".join(char for char in line if char.isprintable() or char == "\t").strip()
        if not line: continue

        # B. Content Cleaning: Strip bullets but PRESERVE keywords and colons
        cleaned = re.sub(r'^[•\*\-\+\•\u2022\u25CF\u25CB\u25AA\u25AB\s\<\>\u00A0\u200B\u200C\u200D\u200E\u200F]+', '', line).strip()
        
        # C. Header Detection (Inclusion/Exclusion)
        header_match = re.match(r'^(Main|Key|Core|Study|Infant|Maternal)?\s*(Inclusion|Exclusion)\s+Criteria', cleaned, re.IGNORECASE)
        if header_match:
            h_type = header_match.group(2).upper()
            header = f"{h_type} CRITERIA"
            if header == last_header: continue
            
            # Header spacing: Double break before new headers
            if processed: 
                while processed and processed[-1] == "": processed.pop()
                processed.append(""); processed.append("") 
            
            processed.append(header)
            last_header = header
            last_line_type = 'HEADER'
            continue
        
        # D. Forensic Fix: Handle keyword headers (e.g., Sex 7 Male -> 7. Sex Male)
        if re.match(r'^(Sex|Reproduction|Age|Informed)\s+\d+', cleaned, re.IGNORECASE):
             cleaned = re.sub(r'^([a-zA-Z]+)\s+(\d+)\s+', r'\2. \1 ', cleaned)
        
        # E. Skip noise lines
        if not cleaned or re.match(r'^[^\w\s]+$', cleaned): continue
            
        # F. Systematic Filter (Marker words)
        noise_markers = ["additional", "none", "n/a", "note", "note:", "main", "key", "core"]
        if cleaned.lower() in noise_markers: continue

        # --- SEMANTIC HIERARCHY ---
        is_starter = re.match(r'^(TITLE|NAME|MEASURE|ENDPOINT|OBJECTIVE):', cleaned, re.IGNORECASE)
        is_meta = re.match(r'^(TIMEFRAME|TIME FRAME|DESC|DESCRIPTION|SYNS|ALIASES|DOSING):', cleaned, re.IGNORECASE)
        
        if is_starter:
            if last_line_type in ['STARTER', 'META', 'GENERIC']:
                while processed and processed[-1] == "": processed.pop()
                processed.append("") 
            
            processed.append(f"• {cleaned}")
            last_line_type = 'STARTER'
        elif is_meta:
            processed.append(f"  {cleaned}")
            last_line_type = 'META'
        elif last_line_type in ['STARTER', 'META']:
            processed.append(f"    {cleaned}")
        else:
            processed.append(f"• {cleaned}")
            last_line_type = 'GENERIC'

    return "\n".join(processed).strip()
