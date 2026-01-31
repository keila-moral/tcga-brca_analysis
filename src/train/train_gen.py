import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.dataset import TCGAPatchDataset
from src.models.model_gen import UNetGenerator, NLayerDiscriminator
from pathlib import Path
import itertools

# Config
BATCH_SIZE = 16
EPOCHS = 80
LR = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints/gen")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def train_gen():
    print(f"Training Generative Model on {DEVICE}")
    
    # Data: split by stage
    # simple hack: load same dataset but filter inside loop or create custom collate
    ds = TCGAPatchDataset(split="train")
    if len(ds) == 0:
        print("No training data found.")
        return
        
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # Models
    # Goal: Late -> Early
    G_L2E = UNetGenerator().to(DEVICE)
    D_E = NLayerDiscriminator().to(DEVICE) # Discriminator for Early domain
    
    optimizer_G = optim.Adam(G_L2E.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D_E.parameters(), lr=LR, betas=(0.5, 0.999))
    
    criterion_GAN = nn.MSELoss()
    criterion_L1 = nn.L1Loss() # Pixel-wise consistency
    
    for epoch in range(EPOCHS):
        loss_Total = torch.tensor(0.0)
        loss_D = torch.tensor(0.0)
        
        for i, (imgs, clinical, _) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            stage = clinical.squeeze().to(DEVICE)
            
            # Create masks
            late_mask = stage >= 3
            early_mask = stage <= 2
            
            real_late = imgs[late_mask]
            real_early = imgs[early_mask]
            
            if len(real_late) == 0 or len(real_early) == 0:
                continue
            
            # --- Train Generator ---
            optimizer_G.zero_grad()
            
            # Generate fake Early from Real Late
            fake_early = G_L2E(real_late)
            
            # Adversarial Loss (Fool D_E)
            pred_fake = D_E(fake_early)
            loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            
            # Identity/Reconstruction Loss (Optional, requires simpler set up)
            # For now, just adversarial + regularizer? 
            # Or simplified: Train G to make Late look like Early without content loss?
            # Actually, without paired data or Cycle consistency, mode collapse is high.
            # But let's assume basic GAN for infra demo.
            loss_Total = loss_GAN
            
            loss_Total.backward()
            optimizer_G.step()
            
            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            
            # Real Loss
            pred_real = D_E(real_early)
            loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            
            # Fake Loss
            pred_fake = D_E(fake_early.detach())
            loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()
            
        print(f"Epoch {epoch}: G Loss {loss_Total.item():.4f}, D Loss {loss_D.item():.4f}")
        
        # Save
        if epoch % 5 == 0:
            torch.save(G_L2E.state_dict(), CHECKPOINT_DIR / f"G_L2E_epoch{epoch}.pth")

if __name__ == "__main__":
    train_gen()
