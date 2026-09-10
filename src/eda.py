"""
Exploratory + explanatory analysis on the parsed HURDAT2 dataset.
Run after src/parse_hurdat2.py. Saves figures to outputs/figures/.

Steps (see PROJECT_BRIEF.md):
  Exploratory: 1) storms/year by category  2) track density map
               3) intensity + RI distribution  4) SW Florida historical set
  Explanatory: 5) Ian's track vs. historical SW FL storms
               6) Ian's intensity curve with RI window flagged
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "hurdat2_clean.csv"
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Sanibel Island, FL
SANIBEL_LAT, SANIBEL_LON = 26.44, -82.10
NAUTICAL_MI_PER_DEG = 60  # approx, good enough for a proximity filter


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    df["category"] = pd.Categorical(
        df["category"], categories=["TD", "TS", "Cat1", "Cat2", "Cat3", "Cat4", "Cat5", "Cat5+"], ordered=True
    )
    return df


def storms_per_year(df: pd.DataFrame):
    """1. Storms per year, and per year at hurricane strength+, over time."""
    per_storm = df.groupby("storm_id").agg(year=("year", "first"), peak_wind=("max_wind_kt", "max"))
    total_by_year = per_storm.groupby("year").size()
    hur_by_year = per_storm[per_storm["peak_wind"] >= 64].groupby("year").size()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(total_by_year.index, total_by_year.values, label="All named/tracked storms", lw=1, alpha=0.7)
    ax.plot(total_by_year.index, hur_by_year.reindex(total_by_year.index, fill_value=0).values,
            label="Hurricane strength (Cat1+)", lw=1.5, color="firebrick")
    ax.plot(total_by_year.index, total_by_year.rolling(10, min_periods=1).mean(),
            "--", color="black", lw=1, label="10yr rolling avg (all storms)")
    ax.set_title("Atlantic Storms per Year, 1851-2025")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of storms")
    ax.legend()
    ax.axvline(1944, color="gray", ls=":", lw=0.8)
    ax.text(1946, ax.get_ylim()[1] * 0.9, "aircraft recon begins", fontsize=8, color="gray")
    ax.axvline(1966, color="gray", ls=":", lw=0.8)
    ax.text(1968, ax.get_ylim()[1] * 0.8, "satellite era begins", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_storms_per_year.png", dpi=150)
    plt.close(fig)


def track_density_map(df: pd.DataFrame):
    """2. Track density map, colored by category, all storms."""
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = {"TD": "#a6a6a6", "TS": "#4daf4a", "Cat1": "#ffff33", "Cat2": "#ff9933",
            "Cat3": "#ff5733", "Cat4": "#e6194b", "Cat5": "#800080", "Cat5+": "#4b0082"}
    for cat, color in cmap.items():
        sub = df[df["category"] == cat]
        ax.scatter(sub["lon"], sub["lat"], s=1, color=color, alpha=0.5)
    ax.scatter([SANIBEL_LON], [SANIBEL_LAT], marker="*", s=250, color="black",
               edgecolor="white", zorder=5)

    # Build legend from fixed-size proxy handles instead of the scatter
    # artists themselves, so markerscale doesn't blow up the Sanibel star.
    category_handles = [
        Line2D([], [], marker="o", linestyle="", color=color, markersize=7, label=cat)
        for cat, color in cmap.items()
    ]
    sanibel_handle = Line2D([], [], marker="*", linestyle="", color="black",
                             markeredgecolor="white", markersize=14, label="Sanibel Island")
    ax.legend(handles=category_handles + [sanibel_handle], loc="upper left", fontsize=8)

    ax.set_xlim(-105, -10)
    ax.set_ylim(0, 60)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Atlantic Basin Track Density by Category, 1851-2025")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_track_density_map.png", dpi=150)
    plt.close(fig)


def intensity_and_ri(df: pd.DataFrame) -> pd.DataFrame:
    """3. Distribution of peak intensity + rapid intensification (RI) events.
    RI = NHC definition: +35kt increase in max wind over a 24hr window."""
    peak = df.groupby("storm_id")["max_wind_kt"].max().dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(peak, bins=30, color="steelblue", edgecolor="white")
    axes[0].set_title("Distribution of Peak Storm Intensity")
    axes[0].set_xlabel("Peak max sustained wind (kt)")
    axes[0].set_ylabel("Number of storms")

    ri_records = []
    for sid, g in df.sort_values("timestamp").groupby("storm_id"):
        g = g.dropna(subset=["max_wind_kt"]).set_index("timestamp")
        if len(g) < 2:
            continue
        wind = g["max_wind_kt"]
        # max wind gain in any trailing 24h window, per observation
        gain = wind - wind.rolling("24h").min()
        max_gain = gain.max()
        ri_records.append({"storm_id": sid, "name": g["name"].iloc[0], "max_24h_gain_kt": max_gain})
    ri_df = pd.DataFrame(ri_records)
    ri_events = ri_df[ri_df["max_24h_gain_kt"] >= 35].sort_values("max_24h_gain_kt", ascending=False)

    axes[1].hist(ri_df["max_24h_gain_kt"].dropna(), bins=30, color="darkorange", edgecolor="white")
    axes[1].axvline(35, color="firebrick", ls="--", label="RI threshold (+35kt/24h)")
    axes[1].set_title("Max 24h Wind Gain per Storm")
    axes[1].set_xlabel("Max wind gain in any 24h window (kt)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_intensity_and_ri.png", dpi=150)
    plt.close(fig)

    print(f"\n{len(ri_events)}/{len(ri_df)} storms ({len(ri_events)/len(ri_df):.1%}) underwent rapid intensification.")
    print("Top 10 RI events on record:")
    print(ri_events.head(10).to_string(index=False))
    return ri_df


def sw_florida_history(df: pd.DataFrame, radius_deg: float = 1.5) -> pd.DataFrame:
    """4. Storms passing within ~radius_deg of Sanibel/Ft. Myers, historically."""
    dist = np.sqrt((df["lat"] - SANIBEL_LAT) ** 2 + (df["lon"] - SANIBEL_LON) ** 2)
    close = df[dist <= radius_deg]
    per_storm = close.groupby("storm_id").agg(
        name=("name", "first"), year=("year", "first"),
        peak_wind_near_sanibel=("max_wind_kt", "max"),
    ).sort_values("peak_wind_near_sanibel", ascending=False)

    print(f"\n{len(per_storm)} storms passed within ~{radius_deg*60:.0f}nm of Sanibel Island since 1851.")
    print("Strongest 15 by wind speed while nearby:")
    print(per_storm.head(15).to_string())
    per_storm.to_csv(ROOT / "data" / "processed" / "sw_florida_storms.csv")
    return per_storm


def ian_deep_dive(df: pd.DataFrame, sw_fl_storms: pd.DataFrame):
    """5 & 6. Ian's track vs. historical SW FL storms, and its intensity curve with RI flagged."""
    ian = df[(df["name"] == "IAN") & (df["year"] == 2022)].sort_values("timestamp")
    if ian.empty:
        print("Hurricane Ian (2022) not found in dataset.")
        return

    # 5. Track overlay
    fig, ax = plt.subplots(figsize=(9, 8))
    for sid in sw_fl_storms.index:
        track = df[df["storm_id"] == sid].sort_values("timestamp")
        ax.plot(track["lon"], track["lat"], color="gray", lw=0.7, alpha=0.5)
    ax.plot(ian["lon"], ian["lat"], color="firebrick", lw=2.5, label="Hurricane Ian (2022)", zorder=5)
    ax.scatter([SANIBEL_LON], [SANIBEL_LAT], marker="*", s=300, color="black",
               edgecolor="white", zorder=6, label="Sanibel Island")
    ax.set_xlim(-90, -70)
    ax.set_ylim(15, 35)
    ax.set_title("Hurricane Ian vs. Historical SW Florida Storm Tracks")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_ian_vs_sw_florida_tracks.png", dpi=150)
    plt.close(fig)

    # 6. Intensity curve with RI window flagged
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(ian["timestamp"], ian["max_wind_kt"], marker="o", color="firebrick", lw=2)
    wind = ian.set_index("timestamp")["max_wind_kt"]
    gain = wind - wind.rolling("24h").min()
    ri_start = gain[gain >= 35].index
    if len(ri_start):
        ax.axvspan(ri_start.min() - pd.Timedelta(hours=24), ri_start.max(), color="orange", alpha=0.25,
                   label="Rapid intensification window (+35kt/24h)")
    landfall = ian[ian["is_landfall"]]
    for _, row in landfall.iterrows():
        ax.axvline(row["timestamp"], color="black", ls="--", lw=1)
        ax.text(row["timestamp"], ax.get_ylim()[1] * 0.95, " landfall", fontsize=8, rotation=90, va="top")
    ax.set_title("Hurricane Ian: Intensity Over Time")
    ax.set_xlabel("Date/Time (UTC)")
    ax.set_ylabel("Max sustained wind (kt)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_ian_intensity_curve.png", dpi=150)
    plt.close(fig)

    print(f"\nIan peak intensity: {ian['max_wind_kt'].max():.0f}kt "
          f"({ian.loc[ian['max_wind_kt'].idxmax(), 'category']})")
    print(f"Landfall point(s):\n{landfall[['timestamp','lat','lon','max_wind_kt','min_pressure_mb']].to_string(index=False)}")


def main():
    df = load()
    storms_per_year(df)
    track_density_map(df)
    intensity_and_ri(df)
    sw_fl = sw_florida_history(df)
    ian_deep_dive(df, sw_fl)
    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
