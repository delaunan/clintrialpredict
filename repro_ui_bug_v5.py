from src.prep.text_cleaning_ui import ui_format_multiline

test_input = """
INCLUSION CRITERIA
• <- Aged 50 to 84 years on the day of inclusion .
EXCLUSION CRITERIA
• -> Participant was pregnant
• --> Participation at the time
"""

output = ui_format_multiline(test_input)
print("--- OUTPUT START ---")
print(output)
print("--- OUTPUT END ---")

if "<-" in output or "->" in output:
    print("BUG: Legacy pointer artifacts remain")
else:
    print("SUCCESS: Legacy pointer artifacts removed")
