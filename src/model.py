"""
Step 4: Predict 24/48/72hr-ahead position and intensity from current state.

For each lead time, trains a RandomForestRegressor per target
(delta-lat, delta-lon, delta-wind) on every storm except Hurricane Ian, which
is held out entirely for the case study in ian_case_study.py.

Evaluation uses GroupKFold (grouped by storm_id) rather than a random split:
a random split would leak information (adjacent 6-hourly observations from
the same storm are highly correlated), which would understate real
out-of-sample error. Track error is reported in nautical miles, combining
the dlat/dlon predictions via great-circle distance, so it's directly
comparable to NHC's own published verification stats.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from features import FEATURE_COLUMNS, LEAD_HOURS
from geo import haversine_nm

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "hurdat2_features.csv"
MODEL_DIR = ROOT / "data" / "processed" / "models"
IAN_STORM_ID = "AL092022"

RF_KWARGS = dict(n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)


def load_training_frame() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"])
    df = df[df["storm_id"] != IAN_STORM_ID]
    return df.dropna(subset=FEATURE_COLUMNS)


def evaluate_lead(df: pd.DataFrame, lead: int) -> dict:
    targets = [f"target_dlat_{lead}h", f"target_dlon_{lead}h", f"target_dwind_{lead}h"]
    rows = df.dropna(subset=targets)
    X = rows[FEATURE_COLUMNS].values
    groups = rows["storm_id"].values
    y_dlat = rows[f"target_dlat_{lead}h"].values
    y_dlon = rows[f"target_dlon_{lead}h"].values
    y_dwind = rows[f"target_dwind_{lead}h"].values

    n_splits = min(5, rows["storm_id"].nunique())
    gkf = GroupKFold(n_splits=n_splits)
    oof_dlat = np.full(len(rows), np.nan)
    oof_dlon = np.full(len(rows), np.nan)
    oof_dwind = np.full(len(rows), np.nan)

    for train_idx, test_idx in gkf.split(X, groups=groups):
        for y, oof in [(y_dlat, oof_dlat), (y_dlon, oof_dlon), (y_dwind, oof_dwind)]:
            m = RandomForestRegressor(**RF_KWARGS)
            m.fit(X[train_idx], y[train_idx])
            oof[test_idx] = m.predict(X[test_idx])

    pred_lat = rows["lat"].values + oof_dlat
    pred_lon = rows["lon"].values + oof_dlon
    actual_lat = rows["lat"].values + y_dlat
    actual_lon = rows["lon"].values + y_dlon
    track_err_nm = haversine_nm(pred_lat, pred_lon, actual_lat, actual_lon)
    intensity_err_kt = np.abs(oof_dwind - y_dwind)

    return {
        "lead_h": lead,
        "n": len(rows),
        "track_err_nm_mean": track_err_nm.mean(),
        "track_err_nm_median": np.median(track_err_nm),
        "intensity_err_kt_mean": intensity_err_kt.mean(),
        "intensity_err_kt_median": np.median(intensity_err_kt),
    }


def fit_final_models(df: pd.DataFrame):
    """Refit on 100% of non-Ian data (no holdout) for use in the Ian case study."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for lead in LEAD_HOURS:
        targets = [f"target_dlat_{lead}h", f"target_dlon_{lead}h", f"target_dwind_{lead}h"]
        rows = df.dropna(subset=targets)
        X = rows[FEATURE_COLUMNS].values
        for target in targets:
            m = RandomForestRegressor(**RF_KWARGS)
            m.fit(X, rows[target].values)
            joblib.dump(m, MODEL_DIR / f"{target}.joblib")
    joblib.dump(FEATURE_COLUMNS, MODEL_DIR / "feature_columns.joblib")
    print(f"Final models saved to {MODEL_DIR}")


def main():
    df = load_training_frame()
    print(f"Training universe: {df['storm_id'].nunique():,} storms, {len(df):,} observations (Ian excluded)\n")

    results = [evaluate_lead(df, lead) for lead in LEAD_HOURS]
    results_df = pd.DataFrame(results)
    print("Cross-validated out-of-sample error (GroupKFold by storm, 5 folds):")
    print(results_df.to_string(index=False))

    # NHC's own published verification (2022 season average, all Atlantic storms).
    # Source: nhc.noaa.gov/verification, 1989-present_OFCL_ATL_annual_trk_errors.pdf
    # and 1990-present_OFCL_ATL_annual_int_errors.pdf
    nhc_2022 = pd.DataFrame({
        "lead_h": [24, 48, 72],
        "nhc_track_err_nm": [31.3, 52.5, 78.1],
        "nhc_intensity_err_kt": [5.8, 7.9, 10.0],
    })
    comparison = results_df.merge(nhc_2022, on="lead_h")
    print("\nVs. NHC's official 2022-season average forecast error (all Atlantic storms):")
    print(comparison[["lead_h", "track_err_nm_mean", "nhc_track_err_nm",
                       "intensity_err_kt_mean", "nhc_intensity_err_kt"]].to_string(index=False))
    print("\nNote: this is a simple kinematic model (no NWP guidance, no human forecaster synthesis,")
    print("no ocean heat content) evaluated the same way NHC verifies its own forecasts (point-by-point")
    print("along the actual track). It is not expected to beat NHC's operational forecast; the comparison")
    print("is meant to show how much of NHC's skill comes from persistence/climatology-style predictors")
    print("alone vs. from the full operational forecast process.")

    fit_final_models(df)


if __name__ == "__main__":
    main()
