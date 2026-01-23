import pandas as pd
import re
import os
from dataclasses import dataclass
from typing import Tuple, Union

@dataclass
class CleanStats:
    original_len: int
    cleaned_len: int
    removed_chars: int
    removed_struck_blocks: int
    removed_admin_spans: int

def day_zero_reconstructor(
    text: str,
    pillar_label: str = "text",
    drop_struck: bool = True,
    strip_html: bool = True,
    return_stats: bool = False,
) -> Union[str, Tuple[str, CleanStats]]:
    """
    Day-zero text cleaner for ClinicalTrials.gov / AACT free-text fields.

    Goals:
      1) Preserve biomedical tokens (e.g., HbA1c, ECOG) and your [SEP] separators.
      2) Remove post-registration edit trails and obvious administrative leakage.
      3) Prevent word-merging and keep a stable tokenization surface for BioBERT.

    Note:
      Text cleaning alone cannot guarantee "day-zero". Use metadata gating as well.
    """
    # Explicit NaN / None handling (robust to pandas CSV parsing)
    if text is None or (isinstance(text, float) and pd.isna(text)):
        cleaned = f"No {pillar_label} provided"
        return (cleaned, CleanStats(0, len(cleaned), 0, 0, 0)) if return_stats else cleaned

    if not isinstance(text, str) or not text.strip() or text.strip().lower() == "nan":
        cleaned = f"No {pillar_label} provided"
        return (cleaned, CleanStats(0, len(cleaned), 0, 0, 0)) if return_stats else cleaned

    original_len = len(text)
    removed_struck_blocks = 0
    removed_admin_spans = 0

    # Normalize newlines early
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Protect custom separator
    text = text.replace("[SEP]", " __SEP_TOKEN__ ")

    # Protect curated biomedical terms (preserve casing)
    protected_terms = ["RECIST", "CTCAE", "NYHA", "ECOG", "HbA1c", "PFS", "OS", "ORR"]
    for i, term in enumerate(protected_terms):
        text = re.sub(rf"(?i)\b{re.escape(term)}\b", f"__PROT_{i}__", text)

    # Remove struck-out content (HTML), because it is almost always “history”
    if drop_struck:
        struck_pattern = r"(?is)<\s*(s|strike)\b[^>]*>.*?<\s*/\s*\1\s*>"
        # Count blocks removed (approx)
        removed_struck_blocks = len(re.findall(struck_pattern, text))
        text = re.sub(struck_pattern, " ", text)

    # Strip other HTML tags but keep content
    if strip_html:
        text = re.sub(r"(?is)<[^>]+>", " ", text)

    # Common AACT / CTG artifacts (replace by spaces to avoid word-merging)
    text = text.replace("~", " ")
    text = text.replace("\\>", " ")
    text = text.replace("*", " ")
    text = re.sub(r"-{2,}", " ", text)

    # Administrative leakage purge
    boundary = r"(?=(?:[.;:\n•\-\u2022]|$))"
    admin_patterns = [
        rf"(?i)\bprotocol\s+v(?:er|ersion)?\.?\s*[\w\.]+\b{boundary}",
        rf"(?i)\bversion\b\s*[\w\.]+\b{boundary}",
        rf"(?i)\b(?:last\s+updated|study\s+record\s+updated)\b.*?{boundary}",
        rf"(?i)\b(?:revised|updated|modified)\b.*?\b(?:on|as\s+of|effective|dated|per|by)\b.*?{boundary}",
        rf"(?i)\bclinicaltrials\.gov\b.*?{boundary}",
        rf"(?i)\bnct\d{{8}}\b.*?{boundary}",
        rf"(?i)\bamendment\b.*?\b(?:protocol|version|dated|date|effective)\b.*?{boundary}",
    ]
    for pat in admin_patterns:
        removed_admin_spans += len(list(re.finditer(pat, text)))
        text = re.sub(pat, " ", text)

    # Restore protected terms and normalize whitespace
    for i, term in enumerate(protected_terms):
        text = text.replace(f"__PROT_{i}__", term)

    text = text.replace("__SEP_TOKEN__", "[SEP]")
    text = re.sub(r"\s+", " ", text).strip()

    cleaned = text if len(text) > 5 else f"No {pillar_label} provided"

    if return_stats:
        stats = CleanStats(
            original_len=original_len,
            cleaned_len=len(cleaned),
            removed_chars=max(0, original_len - len(cleaned)),
            removed_struck_blocks=removed_struck_blocks,
            removed_admin_spans=removed_admin_spans,
        )
        return cleaned, stats

    return cleaned


# ==============================================================================
# PROTECTED EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    input_path = "data/project_data.csv"
    if not os.path.exists(input_path):
        print(f"ERROR: Could not find {input_path}")
    else:
        df = pd.read_csv(input_path, dtype=str)
        print(">>> Executing Day-Zero Scientific Sanitization...")

        pillars = ["txt_scientific_essence", "txt_criteria", "txt_primary_endpoints"]

        # Collect stats only for criteria (efficient + matches your audit file)
        criteria_stats = []

        for col in pillars:
            if col in df.columns:
                label = col.replace("txt_", "")

                # If you only want stats for txt_criteria, do it here in the same pass
                if col == "txt_criteria":
                    cleaned_and_stats = df[col].apply(lambda x: day_zero_reconstructor(x, label, return_stats=True))
                    df[col] = cleaned_and_stats.apply(lambda t: t[0])   # cleaned text

                    # store stats aligned with df rows
                    stats_only = cleaned_and_stats.apply(lambda t: t[1])
                    criteria_stats = stats_only.tolist()

                else:
                    # normal cleaning, no stats
                    df[col] = df[col].apply(lambda x: day_zero_reconstructor(x, label))


        output_path = "data/project_data_nlp_light.csv"
        keep_cols = ["nct_id", "target", "txt_scientific_essence", "txt_criteria", "txt_primary_endpoints"]
        df[keep_cols].to_csv(output_path, index=False)
        print(f"[SUCCESS] File saved: {output_path}")


        print(">>> Saving cleanup stats audit for eligibility criteria...")

        if "txt_criteria" in df.columns and len(criteria_stats) == len(df):
            df_stats = pd.DataFrame([{
                "original_len": s.original_len,
                "cleaned_len": s.cleaned_len,
                "removed_chars": s.removed_chars,
                "removed_admin_spans": s.removed_admin_spans,
                "removed_struck_blocks": s.removed_struck_blocks,
            } for s in criteria_stats])

            if "nct_id" in df.columns:
                df_stats.insert(0, "nct_id", df["nct_id"].values)
            else:
                df_stats.insert(0, "nct_id", pd.Series([None] * len(df)))

            audit_path = "data/criteria_cleaning_stats.csv"
            df_stats.to_csv(audit_path, index=False)
            print(f"[AUDIT] Cleanup stats saved to {audit_path}")
        else:
            print("[AUDIT] Skipped: txt_criteria missing or stats length mismatch.")
