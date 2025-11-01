from __future__ import annotations
from datetime import timedelta
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import numpy as np
from .utils import load_bundle_for_token, FEATURES
from .data_source import fetch_ohlc_kraken, build_exact_training_features


from fastapi import Query
from fastapi.responses import JSONResponse
from .utils import load_bundle_for_token, FEATURES
from .data_source import fetch_ohlc_kraken, build_exact_training_features
import numpy as np
import pandas as pd
import math
import sklearn
from datetime import datetime, timezone
from fastapi import Query


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


@app.get("/debug/snapshot/{pair}", tags=["debug"])
def debug_snapshot(pair: str):
    """Return the last candle, the engineered feature row, and a few sanity stats."""
    token = pair.upper().replace("USD", "")
    bundle = load_bundle_for_token(token)

    df_raw = fetch_ohlc_kraken(pair=pair.upper(), days_back=60)
    X_last, asof_ts = build_exact_training_features(df_raw)

    last_row = df_raw.iloc[-1].to_dict()
    feat_row = {c: float(X_last.iloc[0][c]) for c in X_last.columns}

    stats = {
        "asof_utc": str(asof_ts),
        "last_close": float(df_raw["close"].iloc[-1]),
        "last_high": float(df_raw["high"].iloc[-1]),
        "min_close_60d": float(df_raw["close"].min()),
        "max_close_60d": float(df_raw["close"].max()),
        "feature_minmax": {c: (float(X_last[c].min()), float(X_last[c].max())) for c in X_last.columns},
    }

    meta = bundle.meta or {}
    model_info = {
        "type": type(bundle.model).__name__,
        "is_pipeline": bundle.is_pipeline(),
        "has_feature_names_in_": getattr(bundle.model, "feature_names_in_", None) is not None,
        "scaler_attached": bundle.scaler is not None,
        "target_transform": meta.get("target_transform"),
        "target_scale_divisor": meta.get("target_scale_divisor", 1.0),
    }

    return {
        "asof_ts": str(asof_ts),
        "model_info": model_info,
        "raw_tail": last_row,
        "features_tail": feat_row,
        "stats": stats,
        "features_order_expected": FEATURES,
    }

@app.get("/debug/check_artifacts/{token}", tags=["debug"])
def debug_check_artifacts(token: str):
    """Check model/scaler shapes, feature names, and scaler means."""
    bundle = load_bundle_for_token(token.upper())
    out = {
        "model_type": type(bundle.model).__name__,
        "is_pipeline": bundle.is_pipeline(),
        "model_has_feature_names_in_": getattr(bundle.model, "feature_names_in_", None) is not None,
        "model_feature_names_in_": list(map(str, getattr(bundle.model, "feature_names_in_", []))) if getattr(bundle.model, "feature_names_in_", None) is not None else None,
        "code_FEATURES": FEATURES,
        "scaler_used": bundle.scaler is not None,
    }
    if bundle.scaler is not None:
        sc = bundle.scaler
        out.update({
            "scaler_class": type(sc).__name__,
            "scaler_n_features_in_": getattr(sc, "n_features_in_", None),
            "scaler_mean_head": list(map(float, getattr(sc, "mean_", [])[:8])) if hasattr(sc, "mean_") else None,
            "scaler_scale_head": list(map(float, getattr(sc, "scale_", [])[:8])) if hasattr(sc, "scale_") else None,
        })
    return out

@app.get("/debug/predict_path/{pair}", tags=["debug"])
def debug_predict_path(pair: str):
    """Run prediction in three ways to catch double-scaling or target-transform issues."""
    token = pair.upper().replace("USD", "")
    bundle = load_bundle_for_token(token)
    df_raw = fetch_ohlc_kraken(pair=pair.upper(), days_back=400)
    X_last, asof_ts = build_exact_training_features(df_raw)

    feats = {k: float(X_last.iloc[0][k]) for k in bundle.features}
    x_raw = np.asarray([feats[k] for k in bundle.features], dtype=float).reshape(1, -1)

    # 1) Your normal code path
    try:
        yhat_normal = bundle.predict_from_dict(feats)
    except Exception as e:
        yhat_normal = f"ERROR: {e}"

    # 2) Force "no extra scaling" path: if model is Pipeline, give raw; else scale once
    try:
        if bundle.is_pipeline():
            yhat_pipeline = float(bundle.model.predict(x_raw)[0])
        else:
            xs = bundle.scaler.transform(x_raw)
            yhat_pipeline = float(bundle.model.predict(xs)[0])
    except Exception as e:
        yhat_pipeline = f"ERROR: {e}"

    # 3) If model is Pipeline, test the "WRONG" path (double-scale) to see the effect
    try:
        if bundle.is_pipeline():
            # simulate double scaling: external scaler then pipeline's scaler
            if bundle.scaler is not None:
                xs = bundle.scaler.transform(x_raw)
            else:
                # fabricate a fake identity-like scaling to keep shape
                xs = (x_raw - np.mean(x_raw, axis=1, keepdims=True)) / (np.std(x_raw, axis=1, keepdims=True) + 1e-9)
            yhat_double = float(bundle.model.predict(xs)[0])
        else:
            yhat_double = "N/A (model not a Pipeline)"
    except Exception as e:
        yhat_double = f"ERROR: {e}"

    last_high = float(df_raw["high"].iloc[-1])
    return {
        "asof_ts": str(asof_ts),
        "last_high": last_high,
        "yhat_normal": yhat_normal,
        "yhat_pipeline_only": yhat_pipeline,
        "yhat_double_scaled_sim": yhat_double,
        "ratios_vs_last_high": {
            "normal": (yhat_normal / last_high) if isinstance(yhat_normal, (float, int)) else None,
            "pipeline_only": (yhat_pipeline / last_high) if isinstance(yhat_pipeline, (float, int)) else None,
            "double_scaled": (yhat_double / last_high) if isinstance(yhat_double, (float, int)) else None,
        }
    }
@app.get("/test/{pair}", tags=["debug"])
def test_pair(pair: str):
    token = pair.upper().replace("USD", "")
    bundle = load_bundle_for_token(token)
    df_raw = fetch_ohlc_kraken(pair=pair.upper(), days_back=60)
    X_last, asof_ts = build_exact_training_features(df_raw)
    feats = {k: float(X_last.iloc[0][k]) for k in bundle.features}
    yhat = bundle.predict_from_dict(feats)
    last_close = float(df_raw["close"].iloc[-1])
    last_high  = float(df_raw["high"].iloc[-1])
    return {
        "asof_ts": str(asof_ts),
        "last_close": last_close,
        "last_high": last_high,
        "predicted_next_day_high": float(yhat),
        "diff_pct_vs_last_close": float(100.0 * (yhat - last_close) / max(last_close, 1e-9))
    }


@app.get("/predict/by_date/{pair}", tags=["inference"])
def predict_by_date(
    pair: str,
    date: str = Query(..., description="Date in YYYY-MM-DD (UTC) to use as the feature day"),
):
    """
    Predict the *next* day's HIGH using features from the specified date (UTC).
    Example: /predict/by_date/ETHUSD?date=2025-10-28
    """
    token = pair.upper().replace("USD", "")
    try:
        bundle = load_bundle_for_token(token)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model not found for token '{token}': {e}")

    # Parse date
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    # Fetch plenty of history for indicators
    try:
        df_raw = fetch_ohlc_kraken(pair=pair.upper(), days_back=450)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch Kraken OHLC: {e}")

    # Restrict to candles on or before the requested date
    df_hist = df_raw[df_raw["timeOpen"].dt.date <= target_date.date()].copy()
    if df_hist.empty:
        raise HTTPException(status_code=404, detail=f"No OHLC data on or before {date} (UTC).")

    # Build features; returns (X_last, asof_ts)
    try:
        X_last, asof_ts = build_exact_training_features(df_hist)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature build failed for {date}: {e}")

    # If indicators/rolling windows trimmed rows, the last available feature date may be < requested date
    note = "Exact match" if asof_ts.date() == target_date.date() else f"Adjusted to nearest earlier candle: {asof_ts.date()}"

    # Prepare features in correct order
    try:
        feats = {k: float(X_last.iloc[0][k]) for k in bundle.features}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature alignment error: {e}")

    # Predict (this applies model_meta scaling if present)
    try:
        yhat = float(bundle.predict_from_dict(feats))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {e}")

    # Pull the actual last_high of that asof_ts for context
    try:
        last_row = df_hist.loc[df_hist["timeOpen"] == asof_ts].iloc[0]
        last_high = float(last_row["high"])
        last_close = float(last_row["close"])
    except Exception:
        # Fallback: use tail of df_hist
        last_high = float(df_hist["high"].iloc[-1])
        last_close = float(df_hist["close"].iloc[-1])

    return {
        "requested_date_utc": date,
        "actual_feature_date_utc": asof_ts.strftime("%Y-%m-%d"),
        "note": note,
        "last_close_on_feature_date": last_close,
        "last_high_on_feature_date": last_high,
        "predicted_next_day_high": yhat,
        "diff_pct_vs_last_high": round((yhat - last_high) / max(last_high, 1e-6) * 100.0, 3),
    }