import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import sys
import os

# Setup
sys.path.append(".")
os.makedirs("notebooks/screenshots", exist_ok=True)
DATA_DIR = Path("data")
CLINICAL_PATH = DATA_DIR / "processed/clinical_processed.csv"
PATCHES_DIR = DATA_DIR / "processed/patches"
DEVICE = torch.device("cpu")

from src.models.model_survival import SurvivalCNN
from src.models.model_gen import UNetGenerator
from src.data.dataset import TCGAPatchDataset

def run_exploration():
    print("Running Data Exploration...")
    # Load Data
    if not CLINICAL_PATH.exists():
        print("Clinical data missing.")
        return
        
    df = pd.read_csv(CLINICAL_PATH)
    
    # 1. Survival Curve
    plt.figure(figsize=(10, 6))
    sns.histplot(df['overall_survival_months'].dropna(), bins=30, kde=True)
    plt.title("Overall Survival Distribution (Months)")
    plt.xlabel("Months")
    plt.savefig("notebooks/screenshots/01_survival_dist.png")
    plt.close()
    print("Saved 01_survival_dist.png")
    
    # 2. Tumor Stage
    plt.figure(figsize=(8, 4))
    if 'stage_numeric' in df.columns:
        # Fill NaNs to avoid empty plot
        df['stage_plot'] = df['stage_numeric'].fillna("Unknown")
        sns.countplot(y=df['stage_plot'], order=sorted(df['stage_plot'].unique().astype(str)))
        plt.title("Tumor Stage Count")
        plt.tight_layout()
        plt.savefig("notebooks/screenshots/02_tumor_stage.png")
        plt.close()
        print("Saved 02_tumor_stage.png")
    
    # 3. Sample Patches
    slides = [d for d in PATCHES_DIR.iterdir() if d.is_dir()]
    sample_patches = []
    for slide in slides[:3]: 
        patches = list(slide.glob("*.png"))
        if patches:
            sample_patches.extend(np.random.choice(patches, min(3, len(patches)), replace=False))
            
    if sample_patches:
        plt.figure(figsize=(15, 5))
        for i, p in enumerate(sample_patches):
            if i >= 5: break
            plt.subplot(1, 5, i+1)
            img = Image.open(p)
            plt.imshow(img)
            plt.axis('off')
            plt.title(p.parent.name[:10]+"...")
        plt.tight_layout()
        plt.savefig("notebooks/screenshots/03_sample_patches.png")
        plt.close()
        print("Saved 03_sample_patches.png")

def run_inference():
    print("Running Inference...")
    
    # Load Survival Model
    model_surv = SurvivalCNN().to(DEVICE)
    ckpt_path = Path("checkpoints/survival/best_model.pth")
    if ckpt_path.exists():
        model_surv.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model_surv.eval()
    
    # Load Data
    ds = TCGAPatchDataset(split="val", patch_dir="data/processed/patches", 
                          clinical_file="data/processed/clinical_processed.csv",
                          manifest_file="data/raw/image_manifest.csv")
    
    if len(ds) > 0:
        # Survival Inference
        img, clinical, target = ds[0]
        img_tensor = img.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            risk_score = model_surv(img_tensor)
            
        plt.figure()
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f"Predicted Risk: {risk_score.item():.2f}\nActual: {target[0]:.1f}m")
        plt.axis('off')
        plt.savefig("notebooks/screenshots/04_survival_inference.png")
        plt.close()
        print("Saved 04_survival_inference.png")
        
        # GenAI Inference
        gen = UNetGenerator().to(DEVICE)
        # Find latest checkpoint
        gen_ckpts = list(Path("checkpoints/gen").glob("*.pth"))
        if gen_ckpts:
            latest = sorted(gen_ckpts)[-1]
            gen.load_state_dict(torch.load(latest, map_location=DEVICE))
            gen.eval()
            print(f"Loaded GenAI from {latest.name}")
            
            with torch.no_grad():
                fake_early = gen(img_tensor)
            
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            
            def denorm(t):
                return (t * std + mean).clamp(0, 1)
                
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(denorm(img).permute(1, 2, 0))
            ax[0].set_title("Original")
            ax[0].axis('off')
            
            ax[1].imshow(denorm(fake_early.squeeze()).permute(1, 2, 0))
            ax[1].set_title("Generated (Rewound)")
            ax[1].axis('off')
            
            plt.savefig("notebooks/screenshots/05_genai_result.png")
            plt.close()
            print("Saved 05_genai_result.png")
        else:
            print("No GenAI checkpoints found.")
    else:
        print("Dataset empty, cannot run inference.")

if __name__ == "__main__":
    run_exploration()
    run_inference()
