import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.dataset import TCGAPatchDataset
from src.models.model_survival import SurvivalCNN, cox_ph_loss
import numpy as np
import os
from pathlib import Path

# Config
BATCH_SIZE = 32
EPOCHS = 80
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints/survival")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def train():
    print(f"Training on {DEVICE}")
    
    # Data
    train_ds = TCGAPatchDataset(split="train")
    if len(train_ds) == 0:
        print("No training data found. Waiting for image download?")
        return
        
    val_ds = TCGAPatchDataset(split="val")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # workers=0 for robustness
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = SurvivalCNN(backbone_name="resnet18", pretrained=True).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for i, (imgs, _, targets) in enumerate(train_loader):
            imgs = imgs.to(DEVICE)
            # targets is (time, event)
            times = targets[:, 0].to(DEVICE)
            events = targets[:, 1].to(DEVICE)
            
            if epoch == 0 and i == 0:
                print(f"DEBUG: Batch Events: {events.sum().item()}/{len(events)}")
            
            optimizer.zero_grad()
            risk_scores = model(imgs)
            
            loss = cox_ph_loss(risk_scores, events, times)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_loss:.4f}")
        
        # Validation
        if len(val_loader) > 0:
            val_loss = validate(model, val_loader)
            print(f"Epoch {epoch+1}/{EPOCHS}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
                print("Saved best model.")
    
    print("Training complete.")

def validate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, _, targets in loader:
            imgs = imgs.to(DEVICE)
            times = targets[:, 0].to(DEVICE)
            events = targets[:, 1].to(DEVICE)
            
            risk_scores = model(imgs)
            loss = cox_ph_loss(risk_scores, events, times)
            total_loss += loss.item()
            
    return total_loss / len(loader)

if __name__ == "__main__":
    train()
