from src.prep.text_cleaning_ui import ui_format_multiline

test_input = """Inclusion Criteria:~* Have a visual analog scale (VAS) pain value ≥40 and \<95 during screening.~* Other inclusion stuff...~Exclusion Criteria:~* Heart block...~* QT interval measurement \>450 milliseconds (msec) for male participants..."""

output = ui_format_multiline(test_input)
print("--- OUTPUT START ---")
print(output)
print("--- OUTPUT END ---")

if "95 during screening" in output:
    print("SUCCESS: Clinical range <95 preserved")
else:
    print("BUG: Clinical range was DELETED by HTML stripper")

if "EXCLUSION CRITERIA" in output:
    print("SUCCESS: Exclusion header detected correctly")
else:
    print("BUG: Exclusion header missing (Section Swallowed)")
