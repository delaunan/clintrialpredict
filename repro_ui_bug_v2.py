from src.prep.text_cleaning_ui import ui_format_multiline

test_input = """
INCLUSION CRITERIA
• All Subjects
• Subject must be ≥18 years of age.
• Subjects must have an MBL- positive Gram- negative bacteria (an Enterobacteriaceae and/or Stenotrophomonas maltophilia for which the imipenem or meropenem MIC is ≥ 4 µg/mL).
• Systolic blood pressure (SBP) \\ 40 mmHg;
• Additional


INCLUSION CRITERIA


• cIAI Subjects
• . Subject has at least 1 of the following:
• Traumatic perforation of the intestines, only if operated on \\ 12 hours after diagnosis;
• Additional

EXCLUSION CRITERIA
• All Subjects
• Estimated CrCL ≤15 mL/min.
• Additional

EXCLUSION CRITERIA
• cUTI Subjects
• . Any recent history of trauma.
"""

output = ui_format_multiline(test_input)
print("--- OUTPUT START ---")
print(output)
print("--- OUTPUT END ---")

# Check for the specific forensic fixes
if "\\\\" in output:
    print("BUG: Backslash not converted")
if "< 40 mmHg" in output:
    print("SUCCESS: 'SBP \\ 40' converted to '< 40'")
if output.count("INCLUSION CRITERIA") > 1:
    print("BUG: Redundant INCLUSION CRITERIA headers remain")
if "Additional" in output:
    print("BUG: 'Additional' placeholders remain")
if "• ." in output:
    print("BUG: Leading dot artifacts remain")
