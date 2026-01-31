import torch
import torch.nn as nn
from src.models.model_survival import SurvivalCNN

class MultimodalSurvival(nn.Module):
    def __init__(self, tabular_dim=5):
        """
        Fusion model: Image + Clinical Data.
        """
        super().__init__()
        
        # Image Branch (Pretrained CNN)
        self.image_encoder = SurvivalCNN(backbone_name="resnet18", pretrained=True).backbone
        # Output of backbone is global pooled features (512)
        
        # Tabular Branch
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Fusion Head
        self.fusion_head = nn.Sequential(
            nn.Linear(512 + 32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1) # Risk Score
        )
        
    def forward(self, img, tab):
        img_feat = self.image_encoder(img)
        img_feat = torch.flatten(img_feat, 1) # (B, 512)
        
        tab_feat = self.tabular_encoder(tab)
        
        combined = torch.cat((img_feat, tab_feat), dim=1)
        risk = self.fusion_head(combined)
        return risk

if __name__ == "__main__":
    model = MultimodalSurvival(tabular_dim=4)
    img = torch.randn(2, 3, 224, 224)
    tab = torch.randn(2, 4)
    out = model(img, tab)
    print(f"Fusion Output: {out.shape}")
