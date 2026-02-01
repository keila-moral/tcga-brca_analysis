import math
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.dataset import TCGAPatchDataset
from src.models.model_fusion import MultimodalSurvival
from src.models.model_survival import cox_ph_loss

# -----------------------
# Config
# -----------------------
BATCH_SIZE = 32
EPOCHS = 80

LR_HEAD = 1e-4
LR_UNFROZEN = 1e-5
WEIGHT_DECAY = 1e-4

UNFREEZE_EPOCH = 5              # unfreeze at start of this epoch (0-indexed)
UNFREEZE_LAYER4_ONLY = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = Path("checkpoints/fusion")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"


# -----------------------
# Metrics
# -----------------------
def concordance_index(risk, times, events):
    """
    Harrell's C-index (O(n^2)).
    risk: higher => higher risk (shorter survival)
    times: survival/censor time
    events: 1=event observed, 0=censored
    """
    risk = risk.detach().flatten().cpu()
    times = times.detach().flatten().cpu()
    events = events.detach().flatten().cpu()

    n = len(times)
    concordant = 0.0
    permissible = 0.0
    ties = 0.0

    # Comparable pair: i had an event and occurred before j's time
    for i in range(n):
        if events[i] != 1:
            continue
        ti = times[i]
        ri = risk[i]
        for j in range(n):
            if ti < times[j]:
                permissible += 1
                rj = risk[j]
                if ri > rj:
                    concordant += 1
                elif ri == rj:
                    ties += 1

    if permissible == 0:
        return float("nan")
    return float((concordant + 0.5 * ties) / permissible)


def km_median_survival(times, events):
    """
    Kaplan–Meier median survival time (first time S(t) <= 0.5).
    Returns float('inf') if survival never drops below 0.5.
    """
    # CPU, 1D
    times = times.detach().flatten().cpu()
    events = events.detach().flatten().cpu()

    # Sort by time ascending
    order = torch.argsort(times)
    times = times[order]
    events = events[order]

    unique_times = torch.unique(times)

    n = len(times)
    at_risk = n
    surv = 1.0

    idx = 0
    for t in unique_times:
        # count events and censored at time t
        mask = (times == t)
        d = int(torch.sum(events[mask] == 1).item())
        c = int(torch.sum(events[mask] == 0).item())

        if at_risk > 0:
            if d > 0:
                surv *= (1.0 - d / at_risk)
                if surv <= 0.5:
                    return float(t.item())

        # remove everyone who had event/censor at t
        at_risk -= (d + c)
        idx += int(mask.sum().item())

    return float("inf")


def logrank_test(times, events, group):
    """
    Two-group log-rank test (1 df).
    times: tensor (N,)
    events: tensor (N,)
    group: tensor (N,) values 0/1
    Returns: (chi2_stat, p_value)
    """
    times = times.detach().flatten().cpu()
    events = events.detach().flatten().cpu()
    group = group.detach().flatten().cpu()

    # event times only
    event_times = torch.unique(times[events == 1])
    if len(event_times) == 0:
        return float("nan"), float("nan")

    O1_minus_E1 = 0.0
    V = 0.0

    for t in event_times:
        # at risk just before t: time >= t
        at_risk = (times >= t)
        n1 = int(torch.sum(at_risk & (group == 1)).item())
        n0 = int(torch.sum(at_risk & (group == 0)).item())
        n = n1 + n0
        if n <= 1:
            continue

        # events at t
        at_t = (times == t)
        d1 = int(torch.sum(at_t & (events == 1) & (group == 1)).item())
        d0 = int(torch.sum(at_t & (events == 1) & (group == 0)).item())
        d = d1 + d0
        if d == 0:
            continue

        # expected events in group 1
        e1 = d * (n1 / n)

        # variance (hypergeometric)
        # V = sum( n1*n0*d*(n-d) / (n^2*(n-1)) )
        v = (n1 * n0 * d * (n - d)) / (n * n * (n - 1))

        O1_minus_E1 += (d1 - e1)
        V += v

    if V <= 0:
        return float("nan"), float("nan")

    chi2 = (O1_minus_E1 ** 2) / V

    # For df=1: CDF(x) = erf(sqrt(x/2)); so p = 1 - CDF
    p = 1.0 - math.erf(math.sqrt(chi2 / 2.0))
    return float(chi2), float(p)


def high_low_risk_report(risk, times, events):
    """
    Split by median risk into low/high risk groups and compute:
    - counts/events
    - KM median survival per group
    - log-rank chi2 + p
    Returns a dict with summary.
    """
    risk = risk.detach().flatten().cpu()
    times = times.detach().flatten().cpu()
    events = events.detach().flatten().cpu()

    if len(risk) == 0:
        return {}

    thr = torch.median(risk)
    high = (risk > thr).to(torch.int64)  # 1=high risk, 0=low risk

    n_high = int((high == 1).sum().item())
    n_low = int((high == 0).sum().item())

    e_high = int(((high == 1) & (events == 1)).sum().item())
    e_low = int(((high == 0) & (events == 1)).sum().item())

    # KM median survival
    if n_high > 0:
        med_high = km_median_survival(times[high == 1], events[high == 1])
    else:
        med_high = float("nan")

    if n_low > 0:
        med_low = km_median_survival(times[high == 0], events[high == 0])
    else:
        med_low = float("nan")

    chi2, p = logrank_test(times, events, high)

    return {
        "threshold": float(thr.item()),
        "n_high": n_high,
        "n_low": n_low,
        "events_high": e_high,
        "events_low": e_low,
        "median_surv_high": med_high,
        "median_surv_low": med_low,
        "logrank_chi2": chi2,
        "logrank_p": p,
    }


# -----------------------
# Training utilities
# -----------------------
def build_optimizer(model, lr: float):
    return optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )


def freeze_backbone(model: MultimodalSurvival):
    for p in model.image_encoder.parameters():
        p.requires_grad = False


def unfreeze_backbone(model: MultimodalSurvival, layer4_only: bool = True):
    if layer4_only:
        for p in model.image_encoder.layer4.parameters():
            p.requires_grad = True
    else:
        for p in model.image_encoder.parameters():
            p.requires_grad = True


@torch.no_grad()
def validate_full(model, loader):
    """
    Computes:
    - mean Cox loss over batches (for reference)
    - C-index over the whole validation set
    - high/low risk split report (median split)
    """
    model.eval()

    total_loss = 0.0
    all_risk, all_times, all_events = [], [], []

    for imgs, clinical, targets in loader:
        imgs = imgs.to(DEVICE)
        clinical = clinical.to(DEVICE)

        times = targets[:, 0].to(DEVICE)
        events = targets[:, 1].to(DEVICE)

        risk = model(imgs, clinical).squeeze(1)

        # loss (batch-wise approximation)
        loss = cox_ph_loss(risk.unsqueeze(1), events, times)
        total_loss += float(loss.item())

        all_risk.append(risk)
        all_times.append(times)
        all_events.append(events)

    mean_loss = total_loss / max(1, len(loader))

    all_risk = torch.cat(all_risk, dim=0)
    all_times = torch.cat(all_times, dim=0)
    all_events = torch.cat(all_events, dim=0)

    cidx = concordance_index(all_risk, all_times, all_events)
    hl = high_low_risk_report(all_risk, all_times, all_events)

    return mean_loss, cidx, hl


def train():
    print(f"Training Fusion Model on {DEVICE}")

    train_ds = TCGAPatchDataset(split="train")
    if len(train_ds) == 0:
        print("No training data found.")
        return

    val_ds = TCGAPatchDataset(split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MultimodalSurvival(tabular_dim=1).to(DEVICE)

    # Freeze then fine-tune
    freeze_backbone(model)
    optimizer = build_optimizer(model, lr=LR_HEAD)

    best_cidx = -float("inf")

    for epoch in range(EPOCHS):
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

        # Validation: report loss + C-index + high/low risk split
        if len(val_loader) > 0:
            val_loss, val_cidx, hl = validate_full(model, val_loader)

            print(f"Epoch {epoch+1}/{EPOCHS}, Val Loss: {val_loss:.4f} | Val C-index: {val_cidx:.4f}")

            if hl:
                med_hi = hl["median_surv_high"]
                med_lo = hl["median_surv_low"]
                p = hl["logrank_p"]
                chi2 = hl["logrank_chi2"]
                print(
                    "  High/Low risk (median split): "
                    f"n_high={hl['n_high']} (events={hl['events_high']}), "
                    f"n_low={hl['n_low']} (events={hl['events_low']}), "
                    f"median_surv_high={med_hi}, median_surv_low={med_lo}, "
                    f"log-rank chi2={chi2}, p={p}"
                )

            # Save best model by highest C-index (more meaningful than Cox loss for selection)
            if not math.isnan(val_cidx) and val_cidx > best_cidx:
                best_cidx = val_cidx
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"Saved best model. (best_cidx={best_cidx:.4f})")

    print("Training complete.")
    print(f"Best model path: {BEST_MODEL_PATH}")
    print(f"Best val C-index: {best_cidx:.4f}")


if __name__ == "__main__":
    train()

