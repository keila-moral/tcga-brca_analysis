import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from src.data.dataset import TCGAPatchDataset
from src.models.model_fusion import MultimodalSurvival
from src.models.model_survival import cox_ph_loss

# -----------------------
# Config
# -----------------------
BATCH_SIZE = 32
EPOCHS = 80

LR_HEAD = 1e-4          # LR while backbone frozen (train fusion/tabular head)
LR_UNFROZEN = 1e-5      # smaller LR after unfreezing part of the backbone
WEIGHT_DECAY = 1e-4

UNFREEZE_EPOCH = 5      # unfreeze at start of this epoch (0-indexed)
UNFREEZE_LAYER4_ONLY = True  # if False, unfreezes entire backbone

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = Path("checkpoints/fusion")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"


def build_optimizer(model, lr: float):
    """AdamW over only trainable parameters."""
    return optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY
    )


def freeze_backbone(model: MultimodalSurvival):
    """Freeze entire image encoder (ResNet backbone)."""
    for p in model.image_encoder.parameters():
        p.requires_grad = False


def unfreeze_backbone(model: MultimodalSurvival, layer4_only: bool = True):
    """Unfreeze either last ResNet block (layer4) or entire backbone."""
    if layer4_only:
        # ResNet has layer1..layer4
        for p in model.image_encoder.layer4.parameters():
            p.requires_grad = True
    else:
        for p in model.image_encoder.parameters():
            p.requires_grad = True


def validate(model, loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, clinical, targets in loader:
            imgs = imgs.to(DEVICE)
            clinical = clinical.to(DEVICE)

            times = targets[:, 0].to(DEVICE)
            events = targets[:, 1].to(DEVICE)

            risk_scores = model(imgs, clinical)
            loss = cox_ph_loss(risk_scores, events, times)
            total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


def train():
    print(f"Training Fusion Model on {DEVICE}")

    # Data
    train_ds = TCGAPatchDataset(split="train")
    if len(train_ds) == 0:
        print("No training data found.")
        return

    val_ds = TCGAPatchDataset(split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = MultimodalSurvival(tabular_dim=1).to(DEVICE)

    # (2) Freeze backbone first
    freeze_backbone(model)

    # (3) AdamW with weight decay (train head only at first)
    optimizer = build_optimizer(model, lr=LR_HEAD)

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        # Unfreeze at a chosen epoch and rebuild optimizer so new params train
        if epoch == UNFREEZE_EPOCH:
            unfreeze_backbone(model, layer4_only=UNFREEZE_LAYER4_ONLY)
            optimizer = build_optimizer(model, lr=LR_UNFROZEN)
            what = "layer4" if UNFREEZE_LAYER4_ONLY else "entire backbone"
            print(f"Epoch {epoch+1}: Unfroze {what} and rebuilt optimizer (lr={LR_UNFROZEN}).")

        model.train()
        total_loss = 0.0

        for imgs, clinical, targets in train_loader:
            imgs = imgs.to(DEVICE)
            clinical = clinical.to(DEVICE)

            times = targets[:, 0].to(DEVICE)
            events = targets[:, 1].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            risk_scores = model(imgs, clinical)
            loss = cox_ph_loss(risk_scores, events, times)

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_loss:.4f}")

        # Validation + checkpointing
        if len(val_loader) > 0:
            val_loss = validate(model, val_loader)
            print(f"Epoch {epoch+1}/{EPOCHS}, Val Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"Saved best model. (best_val={best_loss:.4f})")

    print("Training complete.")
    print(f"Best model path: {BEST_MODEL_PATH}")
    print(f"Best val loss: {best_loss:.4f}")


if __name__ == "__main__":
    train()
