import torch
import torch.nn as nn
from torchvision import models

class SurvivalCNN(nn.Module):
    def __init__(self, backbone_name="resnet18", pretrained=True):
        """
        CNN for survival prediction.
        Output: Risk score (scalar) for CoxPH loss.
        """
        super(SurvivalCNN, self).__init__()
        
        # Load Backbone
        if backbone_name == "resnet18":
            self.backbone = models.resnet18(weights='DEFAULT' if pretrained else None)
            in_features = self.backbone.fc.in_features
            # Remove FC
            self.backbone.fc = nn.Identity()
        elif backbone_name == "resnet50":
            self.backbone = models.resnet50(weights='DEFAULT' if pretrained else None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError("Backbone not supported")
            
        # Survival Head
        # Simple linear layer to scalar risk
        self.risk_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1) # Output log hazard
        )
        
    def forward(self, x):
        features = self.backbone(x)
        risk = self.risk_head(features)
        return risk

def cox_ph_loss(risk_scores, events, times):
    """
    Negative Partial Likelihood for Cox Proportional Hazards.
    
    Args:
        risk_scores (torch.Tensor): Predicted log hazards (B, 1).
        events (torch.Tensor): Event indicators, 1=dead, 0=censored (B).
        times (torch.Tensor): Survival times (B).
    """
    # Sort by time descending (critical for efficient calculation)
    # But usually batches are random. 
    # Calculation:
    # Loss = - sum_{i: E_i=1} ( h_i - log( sum_{j: T_j >= T_i} exp(h_j) ) )
    
    # 1. Sort descent
    sorted_indices = torch.argsort(times, descending=True)
    risk_scores = risk_scores[sorted_indices].squeeze()
    events = events[sorted_indices]
    times = times[sorted_indices]
    
    # 2. Calculate logsumexp of risk scores for the risk set
    # Matrix implementation
    # risk_set[i, j] = 1 if T_j >= T_i else 0
    # But simpler: cumulative logsumexp? No, cumulative sum of exp.
    
    exp_risk = torch.exp(risk_scores)
    
    # Cumulative sum of exp_risk from first (highest time) to last (lowest time)
    # cumsum in reverse?
    # T sorted desc: T_0 > T_1 > ...
    # Risk set for i includes all j <= i (since T_j >= T_i is true for j <= i in descending list?)
    # Wait, T_0 is valid for risk set of T_0. T_1 is valid for T_1, but T_0 is also valid for T_1 (lived longer).
    # So risk set at time T_i is {j : T_j >= T_i} = {0, 1, ..., i}
    
    # So we need cumulative sum of exp_risk.
    risk_set_sum = torch.cumsum(exp_risk, dim=0) 
    
    # Log of sum
    log_risk_set_sum = torch.log(risk_set_sum)
    
    # 3. Compute likelihood
    # Only for events
    loss = -torch.sum(events * (risk_scores - log_risk_set_sum))
    
    # Normalize by number of events
    num_events = torch.sum(events)
    if num_events == 0:
        return torch.tensor(0.0, requires_grad=True, device=risk_scores.device)
        
    return loss / num_events

if __name__ == "__main__":
    model = SurvivalCNN()
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")
    
    # Test loss
    times = torch.tensor([10.0, 20.0, 5.0, 15.0])
    events = torch.tensor([1.0, 0.0, 1.0, 1.0])
    loss = cox_ph_loss(out, events, times)
    print(f"Loss: {loss.item()}")
