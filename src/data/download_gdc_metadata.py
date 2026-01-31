import requests
import json
import os
import pandas as pd
from pathlib import Path

# Constants
GDC_API_files = "https://api.gdc.cancer.gov/files"
PROJECT_ID = "TCGA-BRCA"
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def query_gdc_images(limit=10):
    """
    Query GDC API for diagnostic tissue slides (WSI).
    """
    print(f"Querying GDC for {limit} TCGA-BRCA diagnostic slides...")
    
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": PROJECT_ID}},
            {"op": "=", "content": {"field": "data_format", "value": "SVS"}},
            {"op": "=", "content": {"field": "experimental_strategy", "value": "Diagnostic Slide"}},
            {"op": "=", "content": {"field": "access", "value": "open"}} 
        ]
    }

    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.submitter_id,cases.case_id",
        "format": "JSON",
        "size": limit
    }

    response = requests.get(GDC_API_files, params=params)
    data = response.json()
    
    if "data" in data and "hits" in data["data"]:
        hits = data["data"]["hits"]
        print(f"Found {len(hits)} slide images.")
        df = pd.json_normalize(hits)
        output_path = RAW_DIR / "image_manifest.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved image manifest to {output_path}")
        return df
    else:
        print("No data found or API error.")
        return pd.DataFrame()

def query_clinical_data():
    """
    Query GDC API for clinical data text files.
    """
    print("Querying GDC for TCGA-BRCA clinical data...")
    
    clinical_endpoint = "https://api.gdc.cancer.gov/cases"
    
    params = {
        "filters": json.dumps({
            "op": "=", 
            "content": {"field": "project.project_id", "value": PROJECT_ID}
        }),
        # Expanded fields to catch vital status
        "fields": (
            "submitter_id,case_id,diagnoses.days_to_death,diagnoses.days_to_last_follow_up,"
            "diagnoses.vital_status,diagnoses.tumor_stage,demographic.gender,demographic.race,demographic.vital_status"
        ),
        "format": "JSON",
        "size": 1000 
    }
    
    response = requests.get(clinical_endpoint, params=params)
    data = response.json()
    
    if "data" in data and "hits" in data["data"]:
        hits = data["data"]["hits"]
        print(f"Found {len(hits)} clinical records.")
        
        flat_data = []
        for hit in hits:
            demographic = hit.get("demographic", {})
            diagnoses = hit.get("diagnoses", [])
            
            record = {
                "case_id": hit.get("case_id"),
                "submitter_id": hit.get("submitter_id"),
                "gender": demographic.get("gender"),
                "race": demographic.get("race"),
                # Try demographic first
                "vital_status": demographic.get("vital_status")
            }
            
            if diagnoses:
                diag = diagnoses[0]
                # If not in demographic, try diagnosis
                if not record["vital_status"]:
                    record["vital_status"] = diag.get("vital_status")
                
                record["days_to_death"] = diag.get("days_to_death")
                record["days_to_last_follow_up"] = diag.get("days_to_last_follow_up")
                record["tumor_stage"] = diag.get("tumor_stage")
            
            flat_data.append(record)

        df = pd.DataFrame(flat_data)
        output_path = RAW_DIR / "clinical_data.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved clinical data to {output_path}")
        return df
    else:
        print("Error fetching clinical data.")
        return pd.DataFrame()

if __name__ == "__main__":
    query_gdc_images(limit=5)
    query_clinical_data()
