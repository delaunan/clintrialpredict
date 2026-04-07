from src.prep.text_cleaning_ui import ui_format_multiline

test_input = """
INCLUSION CRITERIA
• Aged 18 years or older on the day of inclusion. - -A female participant is eligible to participate if she is not pregnant or breastfeeding... Is of childbearing potential and agrees to use an effective contraceptive method or abstinence.
• Supplemental cohorts, Booster arms: received a complete primary vaccination series with an authorized/conditionally approved mRNA COVID-19 vaccine (mRNA-1273 <[Moderna<] or BNT162b2 <[Pfizer/BioNTech<]) or adenovirus-vectored COVID-19 vaccine (ChAdOx1 nCoV-19 <[Oxford University/AstraZeneca<] or Ad26.CoV2.S <[J<&J/Janssen<]).
"""

output = ui_format_multiline(test_input)
print("--- OUTPUT START ---")
print(output)
print("--- OUTPUT END ---")

# Forensic Checks
if "Moderna<" in output:
    print("BUG: AACT bracket tags (<[) were not cleaned")
if "Aged 18" in output and "A female participant" in output and output.count("•") < 3:
    print("BUG: Double-dash '- -' failed to split the lines")
if "Moderna" in output and "<[" not in output:
    print("SUCCESS: AACT bracket tags cleaned while keeping the word 'Moderna'")
if output.count("•") >= 3:
    print("SUCCESS: '- -' correctly split the criteria into new bullets")
