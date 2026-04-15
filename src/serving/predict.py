# loads the trained LSTM autoencoder and runs inference on a single window
import json
import torch
import numpy as np
from pathlib import Path

# we need the model class definition to load the weights
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.training.model import LSTMAutoencoder


# paths relative to the project root
MODEL_DIR   = Path(__file__).resolve().parents[2] / "data" / "processed" / "model"
CONFIG_PATH = MODEL_DIR / "model_config.json"
WEIGHTS_PATH = MODEL_DIR / "lstm_autoencoder.pt"


def load_model():
    """Load model config, instantiate architecture, load weights."""
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    model = LSTMAutoencoder(
        n_features=config["n_features"],
        hidden_size=config["hidden_size"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
    )

    model.load_state_dict(
        torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model, config


def predict(model, config, window: list[list[float]]) -> dict:
    """
    Run inference on a single window.
    window: list of 12 rows, each row has 7 feature values (already normalized)
    returns: reconstruction error, anomaly flag, threshold used
    """
    x = torch.tensor(np.array(window, dtype=np.float32)).unsqueeze(0)
    # shape: (1, window_size, n_features)

    with torch.no_grad():
        reconstruction = model(x)
        mse = ((reconstruction - x) ** 2).mean().item()

    is_anomaly = int(mse > config["threshold"])

    return {
        "reconstruction_error": round(mse, 8),
        "threshold":            round(config["threshold"], 8),
        "is_anomaly":           is_anomaly,
    }