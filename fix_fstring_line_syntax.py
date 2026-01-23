import nbformat as nbf

def fix_fstring_line_syntax_final():
    file_path = 'notebooks/production_01.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    target_marker = '# --- STEP 13: COMPREHENSIVE TA PERFORMANCE ANALYSIS ---'
    
    # Problematic line (exactly as it appears in the source above)
    old_line = "    line = f'    {row[\'TA\']:<20} | {row[\'N Hist\']:<8} | {row[\'Fail% Hist\']:<10} | {row[\'N Prod\']:<8} | {row[\'Fail% Prod\']:<10} | {row[\'AUC\']:<6} | {row[\'Prec\']:<6} | {row[\'Rec\']:<6} | {row[\'Thresh\']}'"
    # Corrected line using outer double quotes
    new_line = '    line = f"{row[\'TA\']:<20} | {row[\'N Hist\']:<8} | {row[\'Fail% Hist\']:<10} | {row[\'N Prod\']:<8} | {row[\'Fail% Prod\']:<10} | {row[\'AUC\']:<6} | {row[\'Prec\']:<6} | {row[\'Rec\']:<6} | {row[\'Thresh\']}"'

    found = False
    for cell in nb.cells:
        if cell.cell_type == 'code' and target_marker in cell.source:
            if old_line in cell.source:
                cell.source = cell.source.replace(old_line, new_line)
                found = True
                print("Replaced problematic line successfully.")
            else:
                # If exact match fails, try splitting and looking for the line part
                lines = cell.source.split('\n')
                for i, line in enumerate(lines):
                    if "line = f'{" in line and "row['TA']" in line:
                        lines[i] = new_line
                        found = True
                        print(f"Replaced line at index {i} using partial match.")
                cell.source = '\n'.join(lines)
            break

    if not found:
        print("Warning: Could not find the problematic line in the specified cell.")

    with open(file_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Successfully processed {file_path}")

if __name__ == '__main__':
    fix_fstring_line_syntax_final()