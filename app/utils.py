from __future__ import annotations
import os
import joblib
from dataclasses import dataclass
from typing import List
import numpy as np

FEATURES: List[str] = [
    "close", "high", "low", "close_lag1", "open", "MA5", "high_lag1", "low_lag1", "open_lag1",
    "close_lag2", "ICH_Tenkan_9", "high_lag2", "EMA12", "low_lag2", "MA10", "open_lag2",
    "EMA26", "MA20", "BB_mid_20", "ICH_Kijun_26", "BB_up_20", "is_weekend", "month_sin",
    "month_cos", "dow_sin", "dow_cos"
]

@dataclass
class ModelBundle:
    model: object
    scaler: object
    features: List[str]

    def predict_from_dict(self, feats: dict) -> float:
        # ensure all features present
        missing = [f for f in self.features if f not in feats]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # order strictly
        x = np.asarray([float(feats[f]) for f in self.features], dtype=float).reshape(1, -1)
        xs = self.scaler.transform(x)  # raises if feature count mismatches scaler
        return float(self.model.predict(xs)[0])


def _safe_join(*parts: str) -> str:
    return os.path.abspath(os.path.join(*parts))


def load_bundle_for_token(token: str) -> ModelBundle:
    token_dirname = token.upper()
    env_model_dir = os.environ.get("MODEL_DIR")

    if env_model_dir:
        base_models_dir = os.path.abspath(env_model_dir)
    else:
        here = os.path.dirname(__file__)
        repo_root = _safe_join(here, "..")
        base_models_dir = _safe_join(repo_root, "models")

    model_dir = _safe_join(base_models_dir, token_dirname)
    scaler_path = _safe_join(model_dir, "standard_scaler.pkl")
    model_path = _safe_join(model_dir, "lasso_model.pkl")

    if not (os.path.isdir(model_dir) and os.path.isfile(scaler_path) and os.path.isfile(model_path)):
        raise FileNotFoundError(f"Artifacts not found under {model_dir}")

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    return ModelBundle(model=model, scaler=scaler, features=FEATURES)
