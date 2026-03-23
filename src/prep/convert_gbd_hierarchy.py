import pandas as pd
import os

def convert_gbd_hierarchy(input_path, output_dir):
    """
    Robustly converts GBD Hierarchy Excel file into clean CSVs.
    Handles 'Cause Hierarchy' and 'All Location Hierarchies' tabs.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Required tabs mapping: {Tab Name: Output Filename}
    tabs_to_extract = {
        "Cause Hierarchy": "hier_gbd.csv",
        "All Location Hierarchies": "hier_countries.csv"
    }

    print(f"Loading {input_path}...")
    
    try:
        # Using openpyxl as the engine for .xlsx
        xl = pd.ExcelFile(input_path, engine='openpyxl')
        
        # Verify tabs exist
        available_tabs = xl.sheet_names
        print(f"Available tabs: {available_tabs}")

        for tab_name, output_filename in tabs_to_extract.items():
            if tab_name not in available_tabs:
                print(f"Warning: Tab '{tab_name}' not found. Skipping.")
                continue

            print(f"Processing '{tab_name}'...")
            df = pd.read_excel(xl, sheet_name=tab_name)

            # --- Robustness Layer ---
            
            # 1. Strip whitespace from headers
            df.columns = df.columns.astype(str).str.strip()

            # 2. Clean string columns (strip whitespace)
            string_cols = df.select_dtypes(include=['object']).columns
            for col in string_cols:
                df[col] = df[col].astype(str).str.strip()

            # 3. Ensure ID columns are integers (No trailing .0)
            id_cols = [c for c in df.columns if '_id' in c.lower()]
            for col in id_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # We use fillna(0) and astype(int) to ensure clean IDs
                df[col] = df[col].fillna(0).astype(int)

            # 4. Save to CSV
            output_path = os.path.join(output_dir, output_filename)
            df.to_csv(output_path, index=False)
            print(f"Successfully saved: {output_path} ({len(df)} rows)")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Configurable paths
    INPUT_FILE = "data/reference/IHME_GBD_2023_HIERARCHIES.XLSX"
    OUTPUT_FOLDER = "data/reference"
    
    convert_gbd_hierarchy(INPUT_FILE, OUTPUT_FOLDER)
