import re
from typing import Optional

# ------------------------------------------------------------------------------
# STRIKETHROUGH / DELETION PATTERNS (DELETE CONTENT)
# ------------------------------------------------------------------------------

# 1) HTML strikethrough blocks: <s>...</s>, <strike>...</strike>
_STRIKE_HTML_BLOCK = re.compile(
    r"(?is)<\s*(s|strike)\b[^>]*>.*?<\s*/\s*\1\s*>"
)

# 2) Markdown strikethrough: ~~text~~ (non-greedy, multiline-safe)
_STRIKE_MD_BLOCK = re.compile(
    r"(?s)~~.*?~~"
)

# 3) AACT tilde-based strikethrough: ~text~, ~ text ~, multiline-safe
#    This REMOVES EVERYTHING between the first ~ and the next ~
_STRIKE_TILDE_BLOCK = re.compile(
    r"(?s)~\s*.*?\s*~"
)

# ------------------------------------------------------------------------------
# CLEANUP PATTERNS (PRESERVE CONTENT)
# ------------------------------------------------------------------------------

# 4) Any remaining HTML tags (remove tag, keep content)
_HTML_TAGS = re.compile(r"(?is)<[^>]+>")

# 5) Residual artifacts (after struck blocks are removed)
_ARTIFACTS = [
    (re.compile(r"\\" + r">"), " "),   # literal "\>"
    (re.compile(r"\*"), " "),
    (re.compile(r"-{2,}"), " "),       # ---- -> space
]

# 6) Whitespace normalization
_WS = re.compile(r"\s+")

# ------------------------------------------------------------------------------
# MAIN UI CLEANER
# ------------------------------------------------------------------------------

def ui_clean_text(text: Optional[str]) -> str:
    """
    UI-friendly 'latest authoritative text' cleaner.

    Rules:
    - DELETE ALL struck-through content (HTML, Markdown, tilde-based)
    - Strip remaining HTML tags
    - Remove common visual artifacts
    - Normalize whitespace

    This function is intentionally aggressive.
    If it looks deleted in AACT, it is deleted here.
    """

    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    t = text.strip()
    if not t or t.lower() == "nan":
        return ""

    # ------------------------------------------------------------------
    # 1) DELETE ALL STRUCK CONTENT (ORDER MATTERS)
    # ------------------------------------------------------------------
    t = _STRIKE_HTML_BLOCK.sub(" ", t)
    t = _STRIKE_MD_BLOCK.sub(" ", t)
    t = _STRIKE_TILDE_BLOCK.sub(" ", t)

    # ------------------------------------------------------------------
    # 2) REMOVE REMAINING HTML TAGS
    # ------------------------------------------------------------------
    t = _HTML_TAGS.sub(" ", t)

    # ------------------------------------------------------------------
    # 3) REMOVE RESIDUAL ARTIFACTS
    # ------------------------------------------------------------------
    for pat, repl in _ARTIFACTS:
        t = pat.sub(repl, t)

    # ------------------------------------------------------------------
    # 4) NORMALIZE WHITESPACE
    # ------------------------------------------------------------------
    t = _WS.sub(" ", t).strip()

    return t
