
import sys
sys.path.insert(0, "src")

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from dataset import RadioMLDataset, MODULATIONS, load_data, ALL_SNRS
from model import AMCNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_per_snr(model_path="models/amc_all.pt"):
    data  = load_data()
    model = AMCNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    snr_acc = {}
    for snr in ALL_SNRS:
        ds     = RadioMLDataset(data, snr_filter=[snr])
        loader = torch.utils.data.DataLoader(ds, batch_size=512)
        preds, targets = [], []
        with torch.no_grad():
            for x, y in loader:
                preds.extend(model(x.to(DEVICE)).argmax(1).cpu().numpy())
                targets.extend(y.numpy())
        snr_acc[snr] = accuracy_score(targets, preds)

    plt.figure(figsize=(10, 5))
    plt.plot(ALL_SNRS, [snr_acc[s] for s in ALL_SNRS],
             marker="o", linewidth=2, color="steelblue")
    plt.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50% baseline")
    plt.xlabel("SNR (dB)", fontsize=13)
    plt.ylabel("Classification Accuracy", fontsize=13)
    plt.title("AMCNet — Accuracy vs SNR (RadioML 2016.10a)", fontsize=14)
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig("models/accuracy_vs_snr.png", dpi=150)
    plt.show()
    print("Saved accuracy_vs_snr.png")

    ds_high = RadioMLDataset(data, snr_filter=[18])
    loader  = torch.utils.data.DataLoader(ds_high, batch_size=512)
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(model(x.to(DEVICE)).argmax(1).cpu().numpy())
            targets.extend(y.numpy())

    cm = confusion_matrix(targets, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(13, 10))
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                xticklabels=MODULATIONS, yticklabels=MODULATIONS,
                cmap="Blues", vmin=0, vmax=1)
    plt.title("Confusion Matrix at SNR = +18 dB (Normalized)", fontsize=14)
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix_high_snr.png", dpi=150)
    plt.show()
    print("Saved confusion_matrix_high_snr.png")

    print("\nSNR (dB) | Accuracy")
    print("-" * 22)
    for snr in ALL_SNRS:
        print(f"  {snr:+4d}   |  {snr_acc[snr]:.3f}")
