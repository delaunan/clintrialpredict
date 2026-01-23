import nbformat as nbf
import os

def update_beta_for_recall():
    val_nb = 'notebooks/model_validation.ipynb'
    if not os.path.exists(val_nb): return
    
    nb = nbf.read(val_nb, as_version=4)
    print(f"Updating Beta to 1.0 in {val_nb} to increase Recall...")
    
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # 1. Update the Constant in the logic block
            if "F1_calibration = 0.70" in cell.source:
                cell.source = cell.source.replace("F1_calibration = 0.70", "F1_calibration = 1.00 # Balanced Policy")
            
            # 2. Update any hardcoded references in print statements if they exist
            if "Beta=0.70" in cell.source:
                cell.source = cell.source.replace("Beta=0.70", "Beta=1.00")

    nbf.write(nb, val_nb)
    print(f"✅ Successfully updated {val_nb} to Balanced Policy (Beta=1.0)")

if __name__ == '__main__':
    update_beta_for_recall()
