
import re

def ui_format_multiline_v53(text):
    t = str(text).strip()
    # 1. Strip actual tags ONLY
    _HTML_TAGS = re.compile(r"(?is)<(?:[a-z/][^>]*|![^>]*|\[[^>]*\])>")
    t = _HTML_TAGS.sub(" ", t)
    
    # Structural normalization
    t = t.replace("~", "\n").replace(" || ", "\n").replace("- -", "\n")
    
    processed = []
    last_header = ""
    in_semantic_group = False
    
    for line in t.splitlines():
        line = line.strip()
        if not line: continue

        # A. Content Cleaning
        cleaned = re.sub(r'^[•\*\-\+\•\u2022\u25CF\u25CB\u25AA\u25AB\s\:\.\<\> ]+', '', line).strip()
        
        # B. Header Detection
        header_match = re.match(r'^(Main|Key|Core|Study|Infant|Maternal)?\s*(Inclusion|Exclusion)\s+Criteria', cleaned, re.IGNORECASE)
        if header_match:
            h_type = header_match.group(2).upper()
            header = f"{h_type} CRITERIA"
            if header == last_header: continue
            if processed: processed.append("") 
            processed.append(header)
            last_header = header
            in_semantic_group = False
            continue
        
        # Systematic Filter
        noise_markers = ["additional", "none", "n/a", "note", "note:", "main"]
        if cleaned.lower() in noise_markers: continue

        # --- SEMANTIC HIERARCHY ---
        is_starter = re.match(r'^(TITLE|NAME|MEASURE|ENDPOINT):', cleaned, re.IGNORECASE)
        is_meta = re.match(r'^(TIMEFRAME|DESC|SYNS|ALIASES|TIME FRAME):', cleaned, re.IGNORECASE)
        
        if is_starter:
            if processed: processed.append("") 
            processed.append(f"• {cleaned}")
            in_semantic_group = True
        elif is_meta:
            processed.append(f"  {cleaned}")
            in_semantic_group = True
        elif in_semantic_group:
            processed.append(f"    {cleaned}")
        else:
            processed.append(f"• {cleaned}")

    result = "\n".join(processed).strip()
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result

test_input = """TITLE: Maximum Decline From Baseline in Plasma Human Immunodeficiency Virus-1 (HIV-1) Ribonucleic Acid (RNA) Levels - Monotherapy Phase
TIMEFRAME: Baseline (Day 1) and up to Day 84 or end of Monotherapy Phase
TITLE: Number of Participants With Adverse Events (AEs) - Monotherapy Phase
TIMEFRAME: Up to Day 84 or end of Monotherapy Phase
TITLE: Number of Participants With Worst-case Maximum Grade 2-4 Increase in Post-baseline Values of Alanine Aminotransferase (ALT) and Aspartate Aminotransferase (AST) Compared to the Baseline Values - Monotherapy Phase
TIMEFRAME: Baseline (Day 1) and up to Day 84 or end of Monotherapy Phase"""

print(ui_format_multiline_v53(test_input))
