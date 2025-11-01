# app/data_source.py
from __future__ import annotations
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Tuple, List

def _kraken_match_key(result_keys: List[str], pair: str) -> str:
    keys = [k for k in result_keys if k.lower() != "last"]
    up = pair.upper()

    if up in keys:
        return up

    def normalize(k: str) -> str:
        base = k[:-3].upper()
        quote = k[-3:].upper()
        base = base.lstrip("XZ")
        quote = quote.lstrip("XZ")
        return base + quote

    norm_map = {k: normalize(k) for k in keys}
    for k, norm in norm_map.items():
        if norm == up:
            return k

    base, quote = up[:-3], up[-3:]
    for k in keys:
        if base in k.upper() and quote in k.upper():
            return k

    if len(keys) == 1:
        return keys[0]
    raise RuntimeError(f"Pair '{up}' not found. Available keys: {keys}")

def fetch_ohlc_kraken(
    pair: str = "ETHUSD",
    interval: int = 1440,
    days_back: int = 400,   # more history for indicators
) -> pd.DataFrame:
    now_utc = datetime.now(timezone.utc)
    end_date = now_utc.date() - timedelta(days=1)  # yesterday UTC
    start_date = end_date - timedelta(days=days_back)
    since = int(datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc).timestamp())

    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": pair.upper(), "interval": interval, "since": since}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken API error: {data['error']}")

    result = data.get("result", {})
    if not result:
        raise RuntimeError("Kraken returned empty result")

    key = _kraken_match_key(list(result.keys()), pair)
    rows = result[key]

    df = pd.DataFrame(rows, columns=[
        "time", "open", "high", "low", "close", "vwap", "volume", "count"
    ])
    df["timeOpen"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for c in ["open","high","low","close","vwap","volume","count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop(columns=["time"]).dropna().reset_index(drop=True)

    # keep only fully closed candles up to yesterday UTC
    df = df[df["timeOpen"].dt.date <= end_date].reset_index(drop=True)
    return df

def build_exact_training_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Timestamp]:
    df = df.sort_values("timeOpen").reset_index(drop=True)

    # --- Lags ---
    df["close_lag1"] = df["close"].shift(1)
    df["close_lag2"] = df["close"].shift(2)
    df["high_lag1"]  = df["high"].shift(1)
    df["high_lag2"]  = df["high"].shift(2)
    df["low_lag1"]   = df["low"].shift(1)
    df["low_lag2"]   = df["low"].shift(2)
    df["open_lag1"]  = df["open"].shift(1)
    df["open_lag2"]  = df["open"].shift(2)

    # --- MAs ---
    df["MA5"]  = df["close"].rolling(5, min_periods=5).mean()
    df["MA10"] = df["close"].rolling(10, min_periods=10).mean()
    df["MA20"] = df["close"].rolling(20, min_periods=20).mean()

    # --- EMAs ---
    df["EMA12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["close"].ewm(span=26, adjust=False).mean()

    # --- Bollinger (20) ---
    bb_mid = df["close"].rolling(20, min_periods=20).mean()
    bb_std = df["close"].rolling(20, min_periods=20).std()
    df["BB_mid_20"] = bb_mid
    df["BB_up_20"]  = bb_mid + 2 * bb_std

    # --- Ichimoku ---
    tenkan_n, kijun_n = 9, 26
    tenkan = (df["high"].rolling(tenkan_n, min_periods=tenkan_n).max() +
              df["low"].rolling(tenkan_n, min_periods=tenkan_n).min()) / 2.0
    kijun  = (df["high"].rolling(kijun_n, min_periods=kijun_n).max() +
              df["low"].rolling(kijun_n, min_periods=kijun_n).min()) / 2.0
    df["ICH_Tenkan_9"]  = tenkan
    df["ICH_Kijun_26"]  = kijun

    # --- Cyclical + weekend ---
    ts = pd.to_datetime(df["timeOpen"], utc=True, errors="coerce")
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(float)
    month = ts.dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    dow = ts.dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # --- keep ONLY fully-formed rows ---
    keep = [
        "close","high","low","close_lag1","open","MA5","high_lag1","low_lag1","open_lag1",
        "close_lag2","ICH_Tenkan_9","high_lag2","EMA12","low_lag2","MA10","open_lag2",
        "EMA26","MA20","BB_mid_20","ICH_Kijun_26","BB_up_20","is_weekend","month_sin",
        "month_cos","dow_sin","dow_cos","timeOpen"
    ]
    df = df[keep].dropna().reset_index(drop=True)

    # last fully-formed candle = yesterday UTC
    asof_ts = df["timeOpen"].iloc[-1]

    # Safety: ensure prices are positive on the asof row
    row = df.iloc[-1]
    for c in ["open","high","low","close"]:
        if float(row[c]) <= 0:
            raise RuntimeError(f"Non-positive price in {c} at {asof_ts}: {row[c]}")

    X_last = df.drop(columns=["timeOpen"]).iloc[[-1]].astype(float)
    return X_last, asof_ts