import nbformat as nbf

def fix_fstring_syntax():
    file_path = 'notebooks/production_01.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    target_marker = '# --- STEP 13: COMPREHENSIVE TA PERFORMANCE ANALYSIS ---'
    
    # We use a very safe approach: find the line by its prefix and replace it entirely
    # without using complex nested quoting in this python script's source.
    
    for cell in nb.cells:
        if cell.cell_type == 'code' and target_marker in cell.source:
            lines = cell.source.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('header = f\''):
                    # Replace with a version that uses double quotes for the f-string
                    # and single quotes for the keys.
                    new_line = 'header = f"{\'TA\':<20} | {\'Hist N\':<8} | {\'Hist Fail%\':<10} | {\'Prod N\':<8} | {\'Prod Fail%\':<10} | {\'AUC\':<6} | {\'Prec\':<6} | {\'Rec\':<6} | {\'Thresh\'} "'
                    lines[i] = new_line
                    print(f'Fixed header line at index {i}')
            cell.source = '\n'.join(lines)
            break

    with open(file_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Successfully fixed f-string syntax in {file_path}')

if __name__ == '__main__':
    fix_fstring_syntax()
