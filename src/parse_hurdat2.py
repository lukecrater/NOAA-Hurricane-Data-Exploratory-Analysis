"""
Parse NOAA's HURDAT2 best-track file into a tidy DataFrame.

HURDAT2 format (see data/raw/hurdat2-format-atl-1851-2021.pdf):
  Storms are stored as a header line followed by N data lines.

  Header line (4 comma-separated fields):
    AL092022,                IAN,     40,
    [0] basin + ATCF cyclone number + year (storm ID)
    [1] name ("UNNAMED" before 1950)
    [2] number of best-track entries that follow
    [3] empty (trailing comma)

  Data line (21 comma-separated fields):
    20220922, 1800,  , LO, 12.3N,  66.3W,  30, 1006, ... wind radii ..., rmw
    [0] date (YYYYMMDD)
    [1] time (HHMM, UTC)
    [2] record identifier (L=landfall, W=max wind intensity peak, etc; usually blank)
    [3] status of system (HU, TS, TD, EX, LO, SD, SS, WV, DB, ...)
    [4] latitude  (e.g. "28.0N")
    [5] longitude (e.g. "94.8W")
    [6] max sustained wind (knots), -999 if missing
    [7] min central pressure (mb), -999 if missing
    [8:20] 34/50/64-kt wind radii, NE/SE/SW/NW quadrants (post-2004 only)
    [20] radius of max wind (post-2021 only)
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "hurdat2_atlantic.txt"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "hurdat2_clean.csv"

STATUS_LABELS = {
    "TD": "Tropical Depression",
    "TS": "Tropical Storm",
    "HU": "Hurricane",
    "EX": "Extratropical Cyclone",
    "SD": "Subtropical Depression",
    "SS": "Subtropical Storm",
    "LO": "Low",
    "WV": "Tropical Wave",
    "DB": "Disturbance",
}


def _parse_lat(raw: str) -> float:
    raw = raw.strip()
    sign = -1 if raw.endswith("S") else 1
    return sign * float(raw[:-1])


def _parse_lon(raw: str) -> float:
    raw = raw.strip()
    sign = -1 if raw.endswith("W") else 1
    return sign * float(raw[:-1])


def parse_hurdat2(path: Path = RAW_PATH) -> pd.DataFrame:
    rows = []
    storm_id = storm_name = None

    with open(path, "r") as f:
        for line in f:
            fields = [f.strip() for f in line.strip().split(",")]

            if fields[0].startswith(("AL", "EP", "CP")) and len(fields) <= 4:
                storm_id, storm_name = fields[0], fields[1]
                continue

            date, time, record_id, status = fields[0], fields[1], fields[2], fields[3]
            lat, lon, wind, pressure = fields[4], fields[5], fields[6], fields[7]

            rows.append(
                {
                    "storm_id": storm_id,
                    "name": storm_name,
                    "year": int(date[:4]),
                    "timestamp": pd.Timestamp(f"{date} {time}", tz="UTC"),
                    "record_id": record_id or None,
                    "status": status,
                    "status_label": STATUS_LABELS.get(status, status),
                    "lat": _parse_lat(lat),
                    "lon": _parse_lon(lon),
                    "max_wind_kt": int(wind),
                    "min_pressure_mb": int(pressure),
                }
            )

    df = pd.DataFrame(rows)
    df["max_wind_kt"] = df["max_wind_kt"].replace(-999, np.nan)
    df["min_pressure_mb"] = df["min_pressure_mb"].replace(-999, np.nan)
    df["is_landfall"] = df["record_id"] == "L"

    # Saffir-Simpson category from max sustained wind (NaN-safe).
    bins = [-np.inf, 33, 63, 82, 95, 110, 129, 156, np.inf]
    labels = ["TD", "TS", "Cat1", "Cat2", "Cat3", "Cat4", "Cat5", "Cat5+"]
    df["category"] = pd.cut(df["max_wind_kt"], bins=bins, labels=labels)

    df = df.sort_values(["storm_id", "timestamp"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = parse_hurdat2()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Parsed {df['storm_id'].nunique():,} storms, {len(df):,} observations")
    print(f"Years covered: {df['year'].min()}-{df['year'].max()}")
    print(f"Saved to {OUT_PATH}")
