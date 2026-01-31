import pandas as pd
import ast

def check_intersection():
    # Load manifest
    manifest = pd.read_csv("data/raw/image_manifest.csv")
    
    # helper for case parsing
    def get_case(x):
        try:
            if isinstance(x, str):
                c = ast.literal_eval(x)
                if c and isinstance(c, list):
                    return c[0].get('case_id')
            return None
        except:
            return None
            
    manifest['case_id'] = manifest['cases'].apply(get_case)
    image_cases = set(manifest['case_id'].dropna().unique())
    print(f"Image Cases ({len(image_cases)}): {image_cases}")
    
    # Load clinical
    clinical = pd.read_csv("data/processed/clinical_processed.csv")
    clinical_cases = set(clinical['case_id'].unique())
    print(f"Clinical Cases ({len(clinical_cases)}): {list(clinical_cases)[:5]}...")
    
    # Intersection
    common = image_cases.intersection(clinical_cases)
    print(f"Intersection: {len(common)}")
    print(f"Common IDs: {common}")

if __name__ == "__main__":
    check_intersection()
