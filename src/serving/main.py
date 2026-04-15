# FastAPI app — exposes a /predict endpoint for anomaly detection
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

from predict import load_model, predict as run_predict


# --- request / response schemas ---

class PredictRequest(BaseModel):
    # 12 rows x 7 features — already normalized by the caller
    window: list[list[float]]


class PredictResponse(BaseModel):
    reconstruction_error: float
    threshold:            float
    is_anomaly:           int


# --- app lifecycle ---

model = None
config = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # load model once at startup — not on every request
    global model, config
    model, config = load_model()
    print(f"Model loaded — threshold: {config['threshold']:.8f}")
    yield
    # nothing to clean up on shutdown


app = FastAPI(
    title="NAB Anomaly Detection API",
    description="LSTM Autoencoder anomaly detection on streaming time-series windows",
    version="1.0.0",
    lifespan=lifespan
)


# --- endpoints ---

@app.get("/health")
def health():
    """Quick liveness check."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Accept a single normalized window and return reconstruction error
    plus an anomaly flag.

    Expected input: 12 rows, each with 7 values in the order:
    [value, rolling_mean, rolling_std, rolling_zscore,
     rate_of_change, lag_1, lag_2]
    """
    window = request.window

    # basic input validation
    if len(window) != config["window_size"]:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {config['window_size']} rows, got {len(window)}"
        )
    if any(len(row) != config["n_features"] for row in window):
        raise HTTPException(
            status_code=422,
            detail=f"Each row must have {config['n_features']} features"
        )

    result = run_predict(model, config, window)
    return PredictResponse(**result)