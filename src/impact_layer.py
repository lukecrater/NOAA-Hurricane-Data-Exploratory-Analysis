"""
Step 6 (stretch): Impact layer for Hurricane Ian near Sanibel Island.

A full elevation/flood-zone GIS overlay (FEMA flood maps, USGS DEM) would
need a much heavier dependency stack (geopandas/rasterio/shapefiles) and
multi-GB downloads. Instead, this uses NHC's own post-storm survey numbers
from the official Tropical Cyclone Report for Ian (AL092022), which are
arguably more directly relevant than raw elevation contours: they're
observed water levels, not modeled ones.

Source: NHC Tropical Cyclone Report, Hurricane Ian (AL092022), 2023.
data/raw/nhc_verification/AL092022_Ian_TCR.pdf
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "hurdat2_features.csv"
FIG_DIR = ROOT / "outputs" / "figures"
IAN_STORM_ID = "AL092022"
SANIBEL_LAT, SANIBEL_LON = 26.44, -82.10

# Storm surge inundation above ground level (AGL), NHC TCR survey findings.
SURGE_SITES = [
    ("Fort Myers Beach\n(2nd-story marks)", 15.0, "NWS survey, above MHHW"),
    ("Sanibel Island\n(max, eastern end)", 13.0, "USGS/NWS survey, ft AGL"),
    ("Sanibel Island\n(widespread)", 9.0, "USGS survey, ft AGL"),
]
LEE_COUNTY_SURGE_DEATHS = 36
US_TOTAL_SURGE_DEATHS = 41


def load_ian() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"])
    ian = df[df["storm_id"] == IAN_STORM_ID].sort_values("timestamp")
    return ian[ian["timestamp"] >= "2022-09-26"]  # SW Florida approach onward


def plot_impact(ian: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.3, 1]})

    # Panel A: track colored by intensity, zoomed on SW Florida.
    ax = axes[0]
    points = np.array([ian["lon"], ian["lat"]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="YlOrRd", linewidths=4)
    lc.set_array(ian["max_wind_kt"].values[:-1])
    lc.set_clim(60, 140)
    line = ax.add_collection(lc)
    ax.scatter([SANIBEL_LON], [SANIBEL_LAT], marker="*", s=300, color="black",
               edgecolor="white", zorder=5, label="Sanibel Island")
    landfall = ian[ian["is_landfall"]] if "is_landfall" in ian.columns else pd.DataFrame()
    if not landfall.empty:
        fl_landfall = landfall[(landfall["lat"] > 25) & (landfall["lat"] < 28)]
        ax.scatter(fl_landfall["lon"], fl_landfall["lat"], marker="x", s=100, color="black",
                   linewidth=2, zorder=6, label="FL landfall (Cayo Costa)")
    ax.set_xlim(-84, -80)
    ax.set_ylim(25, 28)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Hurricane Ian's Approach to SW Florida")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    cbar = fig.colorbar(line, ax=ax, label="Max sustained wind (kt)", shrink=0.8)

    # Panel B: observed storm surge inundation vs. typical barrier-island elevation.
    ax = axes[1]
    labels = [s[0] for s in SURGE_SITES]
    heights = [s[1] for s in SURGE_SITES]
    colors = ["#08519c", "#e6550d", "#fd8d3c"]
    ax.barh(labels, heights, color=colors)
    for i, (h, site) in enumerate(zip(heights, SURGE_SITES)):
        ax.text(h + 0.2, i, f"{h:.0f} ft", va="center", fontsize=9)
    ax.axvline(3, color="gray", ls="--", lw=1)
    ax.text(3.3, 2.35, "~most barrier-island\nground elevation\n(0-3 ft AGL)", fontsize=7.5, color="gray", va="top")
    ax.set_xlabel("Storm surge inundation (ft above ground level)")
    ax.set_title("Observed Surge vs. Barrier-Island Elevation")
    ax.set_xlim(0, 17)
    ax.set_ylim(-0.6, 2.6)

    fig.suptitle(
        f"Why Sanibel Took the Hit It Did: {LEE_COUNTY_SURGE_DEATHS} of {US_TOTAL_SURGE_DEATHS} US storm-surge deaths "
        "occurred in Lee County, FL",
        y=1.02, fontsize=12,
    )
    fig.text(0.5, -0.02,
              "Source: NHC Tropical Cyclone Report AL092022 (Ian), 2023 — USGS/NWS post-storm surge surveys.",
              ha="center", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_ian_impact_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ian = load_ian()
    plot_impact(ian)
    print(f"Saved {FIG_DIR / '08_ian_impact_layer.png'}")
    print(f"\nContext: NWS/USGS surveys found inundation of 9-15 ft above ground level across")
    print(f"Sanibel/Fort Myers Beach, on barrier islands that mostly sit 0-3 ft above sea level.")
    print(f"Storm surge was Ian's deadliest hazard nationally (41 deaths), {LEE_COUNTY_SURGE_DEATHS} of them in Lee County.")


if __name__ == "__main__":
    main()
