# standalone training script called by the Airflow training DAG
# retrains the LSTM autoencoder on the latest feature store data
# logs everything to MLflow and saves the best model checkpoint
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch

# add project root to path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.training.model import LSTMAutoencoder

# paths
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODEL_DIR     = PROJECT_ROOT / "data" / "processed" / "model"
MLRUNS_DIR    = PROJECT_ROOT / "mlruns"

# hyperparameters
FEATURE_COLS = [
    "value",
    "rolling_mean",
    "rolling_std",
    "rolling_zscore",
    "rate_of_change",
    "lag_1",
    "lag_2"
]
WINDOW_SIZE  = 12
N_FEATURES   = len(FEATURE_COLS)
EPOCHS       = 20
BATCH_SIZE   = 64
LR           = 1e-3
HIDDEN_SIZE  = 64
N_LAYERS     = 2
DROPOUT      = 0.2


class NABWindowDataset(Dataset):
    """Sliding window dataset — same implementation as Notebook 03."""

    def __init__(self, df, window_size, feature_cols):
        self.windows = []
        self.labels  = []

        for series_name, group in df.groupby("series_name"):
            g      = group.sort_values("timestamp").reset_index(drop=True)
            values = g[feature_cols].values.astype(np.float32)
            labels = g["is_anomaly"].values

            for i in range(len(g) - window_size + 1):
                self.windows.append(values[i : i + window_size])
                self.labels.append(int(labels[i : i + window_size].any()))

        self.windows = np.array(self.windows, dtype=np.float32)
        self.labels  = np.array(self.labels,  dtype=np.int64)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx]), torch.tensor(self.labels[idx])


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for windows, _ in loader:
        windows = windows.to(device)
        optimizer.zero_grad()
        reconstruction = model(windows)
        loss = criterion(reconstruction, windows)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(windows)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for windows, _ in loader:
            windows = windows.to(device)
            reconstruction = model(windows)
            loss = criterion(reconstruction, windows)
            total_loss += loss.item() * len(windows)
    return total_loss / len(loader.dataset)


def run_training():
    print(f"Loading features from {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)

    train_df = df[(df["split"] == "train") & (df["is_anomaly"] == 0)].copy()
    val_df   = df[df["split"] == "val"].copy()

    print(f"Train (normal only): {len(train_df):,} rows")
    print(f"Val (all):           {len(val_df):,} rows")

    train_dataset = NABWindowDataset(train_df, WINDOW_SIZE, FEATURE_COLS)
    val_dataset   = NABWindowDataset(val_df,   WINDOW_SIZE, FEATURE_COLS)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model     = LSTMAutoencoder(N_FEATURES, HIDDEN_SIZE, N_LAYERS, DROPOUT).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("lstm_autoencoder_nab")

    with mlflow.start_run(run_name="lstm_retrain") as run:
        mlflow.log_params({
            "epochs":      EPOCHS,
            "batch_size":  BATCH_SIZE,
            "lr":          LR,
            "hidden_size": HIDDEN_SIZE,
            "n_layers":    N_LAYERS,
            "dropout":     DROPOUT,
            "window_size": WINDOW_SIZE,
        })

        best_val_loss = float("inf")

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss   = evaluate(model, val_loader, criterion, device)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss":   val_loss
            }, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(MODEL_DIR, exist_ok=True)
                torch.save(model.state_dict(), MODEL_DIR / "lstm_autoencoder.pt")

            print(f"Epoch {epoch:02d}/{EPOCHS} | "
                  f"train: {train_loss:.6f} | val: {val_loss:.6f}")

        # save threshold as 95th percentile of val reconstruction errors
        model.eval()
        errors = []
        with torch.no_grad():
            for windows, labels in val_loader:
                windows = windows.to(device)
                recon   = model(windows)
                mse     = ((recon - windows) ** 2).mean(dim=(1, 2))
                normal_mask = (labels == 0)
                errors.extend(mse.cpu().numpy()[normal_mask.numpy()])

        threshold = float(np.percentile(errors, 95))
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("threshold",     threshold)
        mlflow.pytorch.log_model(model, name="lstm_autoencoder")

        # update model config with new threshold and run id
        config = {
            "n_features":    N_FEATURES,
            "hidden_size":   HIDDEN_SIZE,
            "n_layers":      N_LAYERS,
            "dropout":       DROPOUT,
            "window_size":   WINDOW_SIZE,
            "feature_cols":  FEATURE_COLS,
            "threshold":     threshold,
            "mlflow_run_id": run.info.run_id,
        }
        with open(MODEL_DIR / "model_config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nTraining complete")
        print(f"Best val loss: {best_val_loss:.6f}")
        print(f"Threshold:     {threshold:.8f}")
        print(f"MLflow run ID: {run.info.run_id}")


if __name__ == "__main__":
    run_training()