from __future__ import annotations
from datetime import timedelta
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import numpy as np

from .utils import load_bundle_for_token, FEATURES
from .data_source import fetch_ohlc_kraken, build_exact_training_features

HORIZON_DAYS = 2  # trained with TARGET_HORIZON = 2

app = FastAPI(
    title="AT3 Crypto HIGH Predictor",
    description="Predict tomorrow's HIGH price using yesterday features (t+2 horizon).",
    version="1.1.0",
)

@app.get("/", response_class=JSONResponse, tags=["info"])
def root():
    return {
        "project": "AMLA AT3 — HIGH price prediction API",
        "objective": "Return the predicted HIGH for the token tomorrow using yesterday features (t+2).",
        "example_endpoint": "/predict/ETHUSD",
        "horizon_note": (
            "The model uses YESTERDAY's closed candle to predict TOMORROW's HIGH (t+2). "
            "All times are UTC and data is fetched live from Kraken."
        ),
        "expected_features_order": FEATURES,
        "github_repo": "https://github.com/KittituchW/AMLA_at3_25544646_FastAPI"
    }

@app.get("/health/", response_class=PlainTextResponse, tags=["health"])
def health():
    return "OK"

@app.get("/predict/{pair}", response_class=JSONResponse, tags=["inference"])
def predict(pair: str):
    token = pair.upper().replace("USD", "")
    try:
        bundle = load_bundle_for_token(token)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model not found for token '{token}': {e}")

    try:
        df_raw = fetch_ohlc_kraken(pair=pair.upper(), days_back=400)
        X_last, asof_ts = build_exact_training_features(df_raw)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch or build features: {e}")

    try:
        feats_dict = {k: float(X_last.iloc[0][k]) for k in bundle.features}
        yhat = bundle.predict_from_dict(feats_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {e}")

    # TEMPORARY safety clamp during debugging (remove once fixed)
    if not np.isfinite(yhat) or yhat <= 0:
        # Fallback to a simple persistence baseline: yesterday's high + small epsilon
        try:
            last_high = float(df_raw.loc[df_raw["timeOpen"] == asof_ts, "high"].iloc[0])
            yhat = max(last_high, 0.01)
        except Exception:
            yhat = 0.01

    return {"predicted_next_day_high": round(float(yhat), 2)}
