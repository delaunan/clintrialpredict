from src.prep.text_cleaning_ui import ui_format_multiline

test_input = """
Inclusion Criteria:
- :
- : Participants who are \>=45 years of age and in generally stable health
- 1.2 kg weight is required
- 1) First bullet
- * Second bullet
- • Third bullet
- Participants who are in generally stable health and have a known diagnosis of Stasis Dermatitis or newly diagnosed Stasis Dermatitis
- Participants whose mental and physical status allows them to be able to mostly perform their activities of daily living with minimal assistance

Exclusion Criteria:

- :
- 1.Participants with clinically significant active or potentially recurrent dermatitis conditions and known genetic dermatological conditions that are not Stasis Dermatitis or overlap with Stasis Dermatitis
- 2) Participants with active venous stasis ulceration on either lower extremity.
- Participants with current infection or suspected infection of any Stasis Dermatitis lesions
- Women of child bearing potential (WOCBP) are not eligible for this study
"""

output = ui_format_multiline(test_input)
print("--- OUTPUT START ---")
print(output)
print("--- OUTPUT END ---")

# Check for the specific issues
if "• :" in output:
    print("BUG DETECTED: '• :' found in output")
if "\n\n\n" in output:
    print("BUG DETECTED: Triple line break found in output")
if "• 2 kg" in output:
    print("BUG DETECTED: '1.2 kg' was incorrectly stripped to '• 2 kg'")
if "• 1.2 kg" in output:
    print("SUCCESS: '1.2 kg' was correctly preserved as '• 1.2 kg'")
