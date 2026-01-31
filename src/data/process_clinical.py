import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
RAW = DATA_DIR / "raw" / "clinical_data.csv"
PROCESSED = DATA_DIR / "processed" / "clinical_processed.csv"

def process_clinical():
    if not RAW.exists():
        print("Raw clinical data not found.")
        return

    df = pd.read_csv(RAW)
    print(f"Loaded {len(df)} clinical records.")

    # Convert days to years (approx) for easier reading
    df['overall_survival_months'] = pd.to_numeric(df['days_to_death'], errors='coerce') / 30.44
    
    # For living patients, use days_to_last_follow_up
    mask_living = df['vital_status'] == 'Alive'
    df.loc[mask_living, 'overall_survival_months'] = \
        pd.to_numeric(df.loc[mask_living, 'days_to_last_follow_up'], errors='coerce') / 30.44

    # Binary outcome: 1 = Dead, 0 = Censored (Alive)
    df['event'] = (df['vital_status'] == 'Dead').astype(int)

    # Drop rows where survival time is missing (cannot use for survival analysis)
    initial_len = len(df)
    df = df.dropna(subset=['overall_survival_months'])
    print(f"Dropped {initial_len - len(df)} records with missing survival time.")

    # Encode Stage
    # Simplified mapping
    stage_map = {
        'stage i': 1, 'stage ia': 1, 'stage ib': 1,
        'stage ii': 2, 'stage iia': 2, 'stage iib': 2,
        'stage iii': 3, 'stage iiia': 3, 'stage iiib': 3, 'stage iiic': 3,
        'stage iv': 4,
        'not reported': np.nan,
        'stage x': np.nan
    }
    df['stage_numeric'] = df['tumor_stage'].astype(str).str.lower().map(stage_map)

    # Save
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED, index=False)
    print(f"Saved processed clinical data to {PROCESSED}")
    print(df[['case_id', 'event', 'overall_survival_months', 'stage_numeric']].head())

if __name__ == "__main__":
    process_clinical()
