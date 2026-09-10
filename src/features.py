"""
Step 3: Feature engineering on top of the parsed HURDAT2 data.

Builds, per observation, lag features describing the storm's recent motion
and intensity trend, plus forward-looking targets (displacement + intensity
change) at 24/48/72hr lead times for the modeling step.

Design choices:
- Restricted to the four standard synoptic times (00/06/12/18 UTC). HURDAT2
  also inserts extra non-synoptic records (landfall, peak intensity, etc.)
  that break the uniform 6-hourly cadence lag/lead features rely on.
- Lag/lead windows are validated by actual elapsed time, not row position,
  so a missing observation doesn't silently pair non-adjacent times.
- Position/intensity targets are stored as *deltas* (change from the current
  observation), not absolute lat/lon/wind, since deltas generalize across
  storms and ocean basins far better than absolute position does.
- Central pressure is available for only ~60% of the historical record
  (pre-satellite-era obs often lack it) and isn't essential for track
  prediction, so it's excluded from the feature set to avoid dropping a
  large fraction of the dataset. Sea-surface-temperature / ocean-heat-content
  data (e.g. NOAA OISST) would meaningfully improve intensity forecasts but
  isn't included here — flagged as a natural v2 addition, not faked with a
  latitude/month proxy.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from geo import bearing_deg, haversine_nm

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "processed" / "hurdat2_clean.csv"
OUT_PATH = ROOT / "data" / "processed" / "hurdat2_features.csv"

LAG_HOURS = [6, 12, 18, 24]
LEAD_HOURS = [24, 48, 72]
SYNOPTIC_TIMES = {"00:00:00", "06:00:00", "12:00:00", "18:00:00"}


def _add_lag(g: pd.DataFrame, hours: int) -> pd.DataFrame:
    steps = hours // 6
    lag_t = g["timestamp"].shift(steps)
    valid = (g["timestamp"] - lag_t) == pd.Timedelta(hours=hours)
    g[f"lat_lag{hours}"] = g["lat"].shift(steps).where(valid)
    g[f"lon_lag{hours}"] = g["lon"].shift(steps).where(valid)
    g[f"wind_lag{hours}"] = g["max_wind_kt"].shift(steps).where(valid)
    return g


def _add_lead(g: pd.DataFrame, hours: int) -> pd.DataFrame:
    steps = hours // 6
    lead_t = g["timestamp"].shift(-steps)
    valid = (lead_t - g["timestamp"]) == pd.Timedelta(hours=hours)
    lead_lat = g["lat"].shift(-steps).where(valid)
    lead_lon = g["lon"].shift(-steps).where(valid)
    lead_wind = g["max_wind_kt"].shift(-steps).where(valid)
    g[f"target_dlat_{hours}h"] = lead_lat - g["lat"]
    g[f"target_dlon_{hours}h"] = lead_lon - g["lon"]
    g[f"target_dwind_{hours}h"] = lead_wind - g["max_wind_kt"]
    return g


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[df["timestamp"].dt.strftime("%H:%M:%S").isin(SYNOPTIC_TIMES)]
    df = df.dropna(subset=["max_wind_kt"])
    df = df.sort_values(["storm_id", "timestamp"])

    out = []
    for storm_id, g in df.groupby("storm_id", sort=False):
        g = g.reset_index(drop=True).copy()
        for h in LAG_HOURS:
            g = _add_lag(g, h)
        for h in LEAD_HOURS:
            g = _add_lead(g, h)
        g["storm_age_hours"] = (g["timestamp"] - g["timestamp"].iloc[0]).dt.total_seconds() / 3600
        out.append(g)
    df = pd.concat(out, ignore_index=True)

    # Motion features from the most recent 6h step.
    df["dlat_6h"] = df["lat"] - df["lat_lag6"]
    df["dlon_6h"] = df["lon"] - df["lon_lag6"]
    df["speed_kt"] = haversine_nm(df["lat_lag6"], df["lon_lag6"], df["lat"], df["lon"]) / 6
    df["heading_deg"] = bearing_deg(df["lat_lag6"], df["lon_lag6"], df["lat"], df["lon"])
    df["heading_sin"] = np.sin(np.radians(df["heading_deg"]))
    df["heading_cos"] = np.cos(np.radians(df["heading_deg"]))

    # Intensity trend features.
    df["dwind_6h"] = df["max_wind_kt"] - df["wind_lag6"]
    df["dwind_24h"] = df["max_wind_kt"] - df["wind_lag24"]

    # Cyclical seasonality.
    month = df["timestamp"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    return df


FEATURE_COLUMNS = [
    "lat", "lon", "max_wind_kt",
    "dlat_6h", "dlon_6h", "speed_kt", "heading_sin", "heading_cos",
    "dwind_6h", "dwind_24h", "storm_age_hours", "month_sin", "month_cos",
]

if __name__ == "__main__":
    raw = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
    feat = build_features(raw)
    feat.to_csv(OUT_PATH, index=False)
    n_full = feat.dropna(subset=FEATURE_COLUMNS + ["dlat_6h", "wind_lag24"]).shape[0]
    print(f"Built features for {len(feat):,} synoptic-time observations ({feat['storm_id'].nunique():,} storms)")
    print(f"{n_full:,} rows have a complete 24h lag window (usable for training)")
    print(f"Saved to {OUT_PATH}")
