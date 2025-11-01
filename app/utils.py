# app/utils.py
from __future__ import annotations
import os, json, joblib, inspect
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

FEATURES: List[str] = [
    "close","high","low","close_lag1","open","MA5","high_lag1","low_lag1","open_lag1",
    "close_lag2","ICH_Tenkan_9","high_lag2","EMA12","low_lag2","MA10","open_lag2",
    "EMA26","MA20","BB_mid_20","ICH_Kijun_26","BB_up_20","is_weekend","month_sin",
    "month_cos","dow_sin","dow_cos"
]

@dataclass
class ModelBundle:
    model: object
    scaler: Optional[object]  # may be None if model is a Pipeline
    features: List[str]
    meta: dict

    def is_pipeline(self) -> bool:
        # sklearn Pipeline duck-typing
        return hasattr(self.model, "named_steps") and hasattr(self.model, "predict")

    def model_feature_names(self) -> Optional[List[str]]:
        return getattr(self.model, "feature_names_in_", None)

    def predict_from_dict(self, feats: dict) -> float:
        missing = [f for f in self.features if f not in feats]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        x_raw = np.asarray([float(feats[f]) for f in self.features], dtype=float).reshape(1, -1)

        # If model is a Pipeline that already contains scaling, don't scale again
        if self.is_pipeline():
            yhat = float(self.model.predict(x_raw)[0])
        else:
            if self.scaler is None:
                raise ValueError("Scaler is None but model is not a Pipeline")
            xs = self.scaler.transform(x_raw)  # raises on mismatch
            yhat = float(self.model.predict(xs)[0])

        # Invert any target transform specified in meta
        tt = (self.meta or {}).get("target_transform", None)
        if tt == "log1p":
            yhat = float(np.expm1(yhat))
        scale = (self.meta or {}).get("target_scale_divisor", 1.0)
        yhat = yhat * float(scale)

        return yhat

def _safe_join(*parts: str) -> str:
    return os.path.abspath(os.path.join(*parts))

def _load_json_if_exists(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def load_bundle_for_token(token: str) -> ModelBundle:
    token_dirname = token.upper()
    env_model_dir = os.environ.get("MODEL_DIR")
    base_models_dir = os.path.abspath(env_model_dir) if env_model_dir else _safe_join(os.path.dirname(__file__), "..", "models")
    model_dir = _safe_join(base_models_dir, token_dirname)

    scaler_path = _safe_join(model_dir, "standard_scaler.pkl")
    model_path  = _safe_join(model_dir, "ridge_model.pkl")
    meta_path   = _safe_join(model_dir, "model_meta.json")

    if not os.path.isdir(model_dir) or not os.path.isfile(model_path):
        raise FileNotFoundError(f"Artifacts not found under {model_dir}")

    model  = joblib.load(model_path)
    meta   = _load_json_if_exists(meta_path)

    scaler = None
    if os.path.isfile(scaler_path):
        scaler = joblib.load(scaler_path)

    # Safety: if model is a Pipeline, ignore external scaler to avoid double-scaling
    if hasattr(model, "named_steps"):
        scaler = None

    # Feature sanity: if model has feature_names_in_, align FEATURES to that
    mfeat = getattr(model, "feature_names_in_", None)
    if mfeat is not None:
        if set(map(str, mfeat)) != set(FEATURES):
            # Hard-fail to catch mismatches early
            raise ValueError(f"Model's feature_names_in_ != FEATURES.\nmodel={list(mfeat)}\ncode ={FEATURES}")

        # Keep code FEATURES order, but you could also reorder to model order if needed:
        # FEATURES = list(map(str, mfeat))  # not mutating the global for safety

    # Optional checks on scaler
    if scaler is not None and hasattr(scaler, "n_features_in_"):
        if scaler.n_features_in_ != len(FEATURES):
            raise ValueError(f"Scaler expects {scaler.n_features_in_} features, got {len(FEATURES)} by code")

    return ModelBundle(model=model, scaler=scaler, features=FEATURES, meta=meta)
