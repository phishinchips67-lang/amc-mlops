
import sys
sys.path.insert(0, "src")

import os
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from dataset import get_dataloaders
from model import AMCNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Credentials come from environment variables set in Cell 1
# No hardcoded values here — safe to push to GitHub
DAGSHUB_USERNAME = os.environ.get("DAGSHUB_USERNAME", "")
DAGSHUB_TOKEN    = os.environ.get("DAGSHUB_TOKEN", "")

if DAGSHUB_USERNAME and DAGSHUB_TOKEN:
    mlflow.set_tracking_uri(
        f"https://dagshub.com/{DAGSHUB_USERNAME}/amc-mlops.mlflow"
    )
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(y)
        correct  += (out.argmax(1) == y).sum().item()
        total    += len(y)
    return loss_sum / total, correct / total

def val_epoch(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out  = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * len(y)
            correct  += (out.argmax(1) == y).sum().item()
            total    += len(y)
    return loss_sum / total, correct / total

def run_experiment(snr_filter=None, epochs=20, lr=1e-3, batch_size=256):
    snr_label = "all" if snr_filter is None else f"{min(snr_filter)}to{max(snr_filter)}"
    print(f"\n=== Experiment: SNR={snr_label} | Device={DEVICE} ===")

    train_loader, val_loader, _ = get_dataloaders(
        snr_filter=snr_filter, batch_size=batch_size
    )
    model     = AMCNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment("AMC-SNR-Study")

    with mlflow.start_run(run_name=f"SNR_{snr_label}"):
        mlflow.log_params({
            "snr_filter": snr_label,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "model": "AMCNet-1DCNN",
            "device": DEVICE,
        })

        for epoch in range(epochs):
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
            vl_loss, vl_acc = val_epoch(model,   val_loader,   criterion)

            mlflow.log_metrics({
                "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss":   vl_loss, "val_acc":   vl_acc,
            }, step=epoch)

            print(f"  Epoch {epoch+1:02d}/{epochs} | "
                  f"train_acc={tr_acc:.3f} | val_acc={vl_acc:.3f}")

        save_path = f"models/amc_{snr_label}.pt"
        torch.save(model.state_dict(), save_path)
        mlflow.log_artifact(save_path)
        mlflow.pytorch.log_model(model, "model")
        print(f"  Model saved → {save_path}")

    return model
