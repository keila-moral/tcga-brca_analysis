import pandas as pd
from pathlib import Path
import ast

def debug_dataset():
    # 1. Load manifest
    manifest_df = pd.read_csv("data/raw/image_manifest.csv")
    manifest_df['slide_id'] = manifest_df['file_name'].apply(lambda x: x.replace('.svs', ''))
    
    def get_case_id(cases_str):
        try:
            if isinstance(cases_str, str):
                cases = ast.literal_eval(cases_str)
            else:
                cases = cases_str
            if isinstance(cases, list) and len(cases) > 0:
                return cases[0].get('case_id')
            return None
        except:
            return None

    manifest_df['case_id'] = manifest_df['cases'].apply(get_case_id)
    
    # 2. Load Clinical
    clinical_df = pd.read_csv("data/processed/clinical_processed.csv")
    
    # 3. Check specific folder
    target_folder = "TCGA-A7-A0CD-01Z-00-DX1.F045B9C8-049C-41BF-8432-EF89F236D34D"
    print(f"Target Slide ID: {target_folder}")
    
    # Check Manifest Map
    row = manifest_df[manifest_df['slide_id'] == target_folder]
    if len(row) == 0:
        print("FAIL: Slide ID not found in manifest.")
    else:
        case_id = row.iloc[0]['case_id']
        print(f"Found Case ID: {case_id}")
        
        # Check Clinical
        clin_row = clinical_df[clinical_df['case_id'] == case_id]
        if len(clin_row) == 0:
            print("FAIL: Case ID not found in clinical data.")
        else:
            print(f"Found Clinical Record: {clin_row.iloc[0].to_dict()}")
            
            # Check Directory
            p = Path("data/processed/patches") / target_folder
            if not p.exists():
                 print("FAIL: Directory does not exist.")
            else:
                 patches = list(p.glob("*.png"))
                 print(f"Found {len(patches)} patches.")

if __name__ == "__main__":
    debug_dataset()
