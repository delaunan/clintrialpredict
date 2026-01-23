import nbformat as nbf
import os

def simplify_to_pure_beta_strategy():
    val_nb = 'notebooks/model_validation.ipynb'
    if not os.path.exists(val_nb): return
    
    nb = nbf.read(val_nb, as_version=4)
    print(f"Simplifying {val_nb} to Pure Beta Strategy (Beta=1.5, No Floor)...")
    
    # 1. Standard F-Beta Function (Simplest Version)
    simple_func = [
        "def find_optimal_threshold(y_true, y_probs, beta=1.0):\n",
        "    ""Standard F-beta maximizer (No Floor)."""\n",
        "    import numpy as np\n",
        "    from sklearn.metrics import precision_recall_curve\n",
        "    p, r, t = precision_recall_curve(y_true, y_probs)\n",
        "    f = (1 + beta**2) * (p * r) / (beta**2 * p + r + 1e-9)\n",
        "    return t[np.argmax(f[:-1])]"
    ]

    for cell in nb.cells:
        if cell.cell_type == 'code':
            # A. Update function definition
            if "def find_optimal_threshold" in "".join(cell.source):
                cell.source = simple_func

            # B. Update Policy Anchor (Set to 1.5 for 70% Recall)
            if "F1_calibration =" in "".join(cell.source):
                cell.source = "# --- POLICY ANCHOR ---\nF1_calibration = 1.50 # Optimized for 70% Recall\n"

            # C. Remove min_prec from function calls
            if "find_optimal_threshold(" in "".join(cell.source):
                cell.source = cell.source.replace(", min_prec=F1_calibration_floor", "")
                cell.source = cell.source.replace(", min_prec=0.40", "")

    nbf.write(nb, val_nb)
    print(f"✅ Successfully simplified {val_nb}. Please re-run the Calibration and Audit cells.")

if __name__ == '__main__':
    simplify_to_pure_beta_strategy()
