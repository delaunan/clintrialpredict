from src.prep.text_cleaning_ui import ui_format_multiline

test_input = """Inclusion Criteria:~* Medically stable on the basis of physical examination...~* Fibroscan liver stiffness measurement less than and equal to (\<=) 9 Kilopascal (kPa) within 6 months prior to screening or at the time of screening~Exclusion Criteria:~* Evidence of infection with hepatitis A, C, D or E...~* Presence of hemoglobinopathy (including sickle cell disease, thalassemia)"""

output = ui_format_multiline(test_input)
print("--- OUTPUT START ---")
print(output)
print("--- OUTPUT END ---")

if "EXCLUSION CRITERIA" in output:
    print("SUCCESS: Exclusion header detected correctly")
else:
    print("BUG: Exclusion header still missing")
