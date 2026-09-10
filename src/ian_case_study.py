"""
Step 5: Hurricane Ian case study.

Applies the models trained in model.py (which never saw Ian) to Ian's own
observations: at every synoptic time along Ian's real track, use the state
at that time to predict position/intensity 24/48/72h ahead, then compare to
what actually happened. This point-by-point-along-the-track methodology
mirrors how NHC verifies its own operational forecasts, which is what makes
the comparison in model.py's NHC benchmark meaningful here too.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS, LEAD_HOURS
from geo import haversine_nm

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "hurdat2_features.csv"
MODEL_DIR = ROOT / "data" / "processed" / "models"
FIG_DIR = ROOT / "outputs" / "figures"
IAN_STORM_ID = "AL092022"

# NHC's official 2022-season average forecast error (all Atlantic storms).
# Source: nhc.noaa.gov/verification (see model.py for the underlying PDFs).
NHC_2022 = pd.DataFrame({
    "lead_h": [24, 48, 72],
    "nhc_track_err_nm": [31.3, 52.5, 78.1],
    "nhc_intensity_err_kt": [5.8, 7.9, 10.0],
})


def load_ian() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"])
    return df[df["storm_id"] == IAN_STORM_ID].dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def predict_lead(ian: pd.DataFrame, lead: int) -> pd.DataFrame:
    m_dlat = joblib.load(MODEL_DIR / f"target_dlat_{lead}h.joblib")
    m_dlon = joblib.load(MODEL_DIR / f"target_dlon_{lead}h.joblib")
    m_dwind = joblib.load(MODEL_DIR / f"target_dwind_{lead}h.joblib")

    X = ian[FEATURE_COLUMNS].values
    pred_dlat = m_dlat.predict(X)
    pred_dlon = m_dlon.predict(X)
    pred_dwind = m_dwind.predict(X)

    out = ian[["timestamp", "lat", "lon", "max_wind_kt"]].copy()
    out["lead_h"] = lead
    out["pred_lat"] = ian["lat"] + pred_dlat
    out["pred_lon"] = ian["lon"] + pred_dlon
    out["pred_wind_kt"] = ian["max_wind_kt"] + pred_dwind
    out["actual_target_time"] = ian["timestamp"] + pd.Timedelta(hours=lead)
    out["actual_lat"] = ian["lat"] + ian[f"target_dlat_{lead}h"]
    out["actual_lon"] = ian["lon"] + ian[f"target_dlon_{lead}h"]
    out["actual_wind_kt"] = ian["max_wind_kt"] + ian[f"target_dwind_{lead}h"]
    out = out.dropna(subset=["actual_lat", "actual_lon", "actual_wind_kt"])

    out["track_err_nm"] = haversine_nm(out["pred_lat"], out["pred_lon"], out["actual_lat"], out["actual_lon"])
    out["intensity_err_kt"] = (out["pred_wind_kt"] - out["actual_wind_kt"]).abs()
    return out


def summarize(all_preds: pd.DataFrame) -> pd.DataFrame:
    summary = all_preds.groupby("lead_h").agg(
        n=("track_err_nm", "size"),
        our_track_err_nm=("track_err_nm", "mean"),
        our_intensity_err_kt=("intensity_err_kt", "mean"),
    ).reset_index()
    return summary.merge(NHC_2022, on="lead_h")


def plot_error_comparison(summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(summary))
    width = 0.35

    axes[0].bar(x - width / 2, summary["our_track_err_nm"], width, label="Our model (on Ian)", color="steelblue")
    axes[0].bar(x + width / 2, summary["nhc_track_err_nm"], width, label="NHC official (2022 avg, all storms)", color="firebrick")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{h}h" for h in summary["lead_h"]])
    axes[0].set_ylabel("Mean track error (nm)")
    axes[0].set_title("Track Error: Our Model vs. NHC Official")
    axes[0].legend()

    axes[1].bar(x - width / 2, summary["our_intensity_err_kt"], width, label="Our model (on Ian)", color="steelblue")
    axes[1].bar(x + width / 2, summary["nhc_intensity_err_kt"], width, label="NHC official (2022 avg, all storms)", color="firebrick")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{h}h" for h in summary["lead_h"]])
    axes[1].set_ylabel("Mean intensity error (kt)")
    axes[1].set_title("Intensity Error: Our Model vs. NHC Official")
    axes[1].legend()

    fig.suptitle("Hurricane Ian Case Study: Simple Kinematic Model vs. NHC's Operational Forecast", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_ian_model_vs_nhc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predicted_tracks(all_preds: pd.DataFrame, ian: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharex=True, sharey=True)
    colors = {24: "#377eb8", 48: "#ff7f00", 72: "#e41a1c"}
    for ax, lead in zip(axes, LEAD_HOURS):
        ax.plot(ian["lon"], ian["lat"], color="black", lw=2, label="Ian's actual track", zorder=3)
        sub = all_preds[all_preds["lead_h"] == lead]
        for _, row in sub.iterrows():
            ax.plot([row["lon"], row["pred_lon"]], [row["lat"], row["pred_lat"]],
                    color=colors[lead], lw=0.7, alpha=0.6, zorder=2)
        ax.scatter(sub["pred_lon"], sub["pred_lat"], s=12, color=colors[lead],
                   label=f"{lead}h-ahead predictions", zorder=4)
        ax.set_title(f"{lead}hr Lead Time")
        ax.set_xlabel("Longitude")
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("Latitude")
    fig.suptitle("Hurricane Ian: Predicted vs. Actual Position at Each Lead Time", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_ian_predicted_tracks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ian = load_ian()
    print(f"Ian: {len(ian)} synoptic observations with a complete 24h lag window")

    all_preds = pd.concat([predict_lead(ian, lead) for lead in LEAD_HOURS], ignore_index=True)
    summary = summarize(all_preds)
    print("\nOur model's error on Ian vs. NHC's 2022 official average (all Atlantic storms):")
    print(summary.to_string(index=False))

    plot_error_comparison(summary)
    plot_predicted_tracks(all_preds, ian)
    all_preds.to_csv(ROOT / "data" / "processed" / "ian_predictions.csv", index=False)
    print(f"\nFigures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
