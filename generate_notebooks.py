import json
import os

def create_notebook(filename, cells):
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    for cell_type, source in cells:
        notebook["cells"].append({
            "cell_type": cell_type,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")],
            "execution_count": None
        })
        
    with open(filename, 'w') as f:
        json.dump(notebook, f, indent=4)
    print(f"Created {filename}")

# --- Notebook 1: Data Exploration ---
nb1_cells = [
    ("markdown", "# TCGA-BRCA Data Exploration\nThis notebook visualizes the dataset statistics, survival curves, and sample image patches."),
    ("code", """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import numpy as np

# Config
DATA_DIR = Path("../data")
CLINICAL_PATH = DATA_DIR / "processed/clinical_processed.csv"
PATCHES_DIR = DATA_DIR / "processed/patches"
"""),
    ("markdown", "## 1. Clinical Data Analysis"),
    ("code", """# Load Data
df = pd.read_csv(CLINICAL_PATH)
print(f"Total Patients: {len(df)}")
df.head()
"""),
    ("code", """# Survival Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['overall_survival_months'].dropna(), bins=30, kde=True)
plt.title("Overall Survival Distribution (Months)")
plt.xlabel("Months")
plt.show()
"""),
    ("code", """# Tumor Stage Distribution
plt.figure(figsize=(8, 4))
sns.countplot(y=df['stage_numeric'], order=sorted(df['stage_numeric'].unique()))
plt.title("Tumor Stage Count")
plt.show()
"""),
    ("markdown", "## 2. Image Patch Visualization\nLet's visualize random patches from the processed slides."),
    ("code", """# Find all patches
slides = [d for d in PATCHES_DIR.iterdir() if d.is_dir()]
print(f"Found {len(slides)} slides.")

# Collect sample patches
sample_patches = []
for slide in slides[:3]: # Look at first 3 slides
    patches = list(slide.glob("*.png"))
    if patches:
        # Pick 3 random
        sample_patches.extend(np.random.choice(patches, min(3, len(patches)), replace=False))

# Plot
if sample_patches:
    plt.figure(figsize=(15, 5))
    for i, p in enumerate(sample_patches):
        if i >= 5: break
        plt.subplot(1, 5, i+1)
        img = Image.open(p)
        plt.imshow(img)
        plt.axis('off')
        plt.title(p.parent.name[:10]+"...")
    plt.show()
else:
    print("No patches found yet.")
""")
]

# --- Notebook 2: Model Inference ---
nb2_cells = [
    ("markdown", "# Model Inference & GenAI Demo\nDemonstrating the trained Survival Prediction model and the Generative AI model."),
    ("code", """import torch
import sys
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append("..")

from src.models.model_survival import SurvivalCNN
from src.models.model_gen import UNetGenerator
from src.data.dataset import TCGAPatchDataset

DEVICE = torch.device("cpu") # Use CPU for demo inference
"""),
    ("markdown", "## 1. Survival Prediction (Phase 1)"),
    ("code", """# Load Model
model_surv = SurvivalCNN().to(DEVICE)
ckpt_path = Path("../checkpoints/survival/best_model.pth")

if ckpt_path.exists():
    model_surv.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model_surv.eval()
    print("Survival Model Loaded.")
else:
    print("Checkpoint not found. Run pipeline first.")
"""),
    ("code", """# Run Inference on a Sample
ds = TCGAPatchDataset(split="val", patch_dir="../data/processed/patches", 
                      clinical_file="../data/processed/clinical_processed.csv",
                      manifest_file="../data/raw/image_manifest.csv")

if len(ds) > 0:
    img, clinical, target = ds[0]
    img_tensor = img.unsqueeze(0).to(DEVICE) # Add batch dim
    
    with torch.no_grad():
        risk_score = model_surv(img_tensor)
        
    print(f"Predicted Risk Score: {risk_score.item():.4f}")
    print(f"Actual Time: {target[0]:.1f} months, Event: {target[1]}")
    
    plt.imshow(img.permute(1, 2, 0)) # CHW -> HWC
    plt.title(f"Risk: {risk_score.item():.2f}")
    plt.axis('off')
    plt.show()
else:
    print("Dataset empty.")
"""),
    ("markdown", "## 2. Generative AI: Rewinding Disease (Phase 3)\nTranslating Late Stage images to Early Stage appearance."),
    ("code", """# Load GenAI Model
gen = UNetGenerator().to(DEVICE)
ckpt_gen_path = Path("../checkpoints/gen/G_L2E_epoch15.pth") # Try to load a later epoch

if not ckpt_gen_path.exists():
    # Fallback to any saved
    avail = list(Path("../checkpoints/gen").glob("*.pth"))
    if avail:
        ckpt_gen_path = sorted(avail)[-1]

if ckpt_gen_path.exists():
    gen.load_state_dict(torch.load(ckpt_gen_path, map_location=DEVICE))
    gen.eval()
    print(f"GenAI Model Loaded from {ckpt_gen_path.name}")
else:
    print("GenAI Checkpoint not found.")
"""),
    ("code", """# Visualize Translation
if len(ds) > 0:
    # Pick a random sample
    idx = np.random.randint(0, len(ds))
    real_img, _, _ = ds[idx]
    
    real_tensor = real_img.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        fake_early = gen(real_tensor)
        
    # Plot Side by Side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Unnormalize for display if needed, but for now assuming roughly [0,1] or standard
    # The transforms used Normalize mean/std, so we should denormalize to look good
    def denorm(t):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (t * std + mean).clamp(0, 1)

    ax[0].imshow(denorm(real_img).permute(1, 2, 0))
    ax[0].set_title("Input (Original)")
    ax[0].axis('off')
    
    ax[1].imshow(denorm(fake_early.squeeze()).permute(1, 2, 0))
    ax[1].set_title("Generated (Early Stage)")
    ax[1].axis('off')
    
    plt.show()
""")
]

if __name__ == "__main__":
    os.makedirs("notebooks", exist_ok=True)
    create_notebook("notebooks/01_Data_Exploration.ipynb", nb1_cells)
    create_notebook("notebooks/02_Model_Inference.ipynb", nb2_cells)
