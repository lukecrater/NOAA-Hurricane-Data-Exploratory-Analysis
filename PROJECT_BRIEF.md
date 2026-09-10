# Hurricane Track & Intensity Forecasting — A Post-Ian Analysis

## Brief

This project uses NOAA's HURDAT2 best-track database to build a model that
forecasts a tropical cyclone's future position and intensity from its current
track history. As a Sanibel Island resident who lived through Hurricane Ian,
I use Ian as a case study — training on all prior storms, then evaluating how
well the model would have predicted Ian's rapid intensification and landfall
track compared to the official NHC forecast issued at the time.

## Data source

- **HURDAT2** (Atlantic hurricane database, 1851–2025): `data/raw/hurdat2_atlantic.txt`
- Format spec: `data/raw/hurdat2-format-atl-1851-2021.pdf`
- Parsed, tidy version: `data/processed/hurdat2_clean.csv` (built by `src/parse_hurdat2.py`)

HURDAT2 is the one dataset on the NHC data page worth pulling directly — it's
clean six-hourly position/wind/pressure data for every storm on record. The
rest of the NHC data page (PDFs, scanned advisory wallets, text bulletins)
isn't structured enough to model against and was skipped.

## Outline

1. **Data ingestion** — parse HURDAT2's fixed-width/CSV-hybrid format into a
   clean DataFrame (storm ID, timestamp, lat/lon, max wind, pressure, status).
   → `src/parse_hurdat2.py`
2. **EDA** — seasonal patterns, track density by region, intensity
   distributions, historical Gulf/SW Florida landfalls. → `src/eda.py`
3. **Feature engineering** — lag features (position/speed/heading over the
   prior 6–24hrs), rate of intensification, distance to warm water.
4. **Modeling** — predict 24/48/72hr-ahead position and intensity from
   current state (regression or simple ML, not full physics).
5. **Case study: Hurricane Ian** — hold Ian out of training, run the model
   against it, and compare predicted track/intensity to the actual NHC
   forecast track at matching lead times. This is the centerpiece: "built a
   model and benchmarked it against NHC's own forecast," not just "built a
   model."
6. **(Stretch) Impact layer** — overlay Ian's actual path against Lee County
   elevation/flood-zone data to visualize why Sanibel took the hit it did.

## Status

- [x] Step 1 — parser built and validated (`src/parse_hurdat2.py`)
- [x] Step 2 — exploratory + explanatory EDA (`src/eda.py`, figures in `outputs/figures/`)
- [x] Step 3 — feature engineering: lag motion/intensity features + 24/48/72h targets (`src/features.py`)
- [x] Step 4 — modeling: RandomForest per lead time, GroupKFold CV, Ian excluded from training (`src/model.py`)
- [x] Step 5 — Ian case study vs. NHC's official 2022-season average forecast error (`src/ian_case_study.py`)
- [x] Step 6 — impact layer (stretch): observed storm surge vs. barrier-island elevation, from NHC's Ian TCR (`src/impact_layer.py`)

### Key results

- **Model performance**: a simple kinematic-only model (position/speed/heading/intensity trend, no NWP
  guidance) trails NHC's official forecast by roughly 3x on track error and 2-3x on intensity error at all
  lead times — both in general cross-validation and specifically on Hurricane Ian. This quantifies how much
  of NHC's forecast skill comes from atmospheric model guidance and human forecaster synthesis, not just
  persistence/climatology-style signal. See `outputs/figures/06_ian_model_vs_nhc.png`.
- **Ian's rapid intensification** (+110kt in the 24h before landfall) is the single hardest event for the
  model to anticipate — intensity error on Ian is proportionally worse than the general holdout error,
  consistent with RI events being notoriously hard to forecast without ocean heat content data.
- **Impact layer**: NHC's post-storm surveys found 9-15ft of storm-surge inundation (above ground level)
  across Sanibel/Fort Myers Beach — on barrier islands that mostly sit 0-3ft above sea level. Storm surge
  was Ian's deadliest hazard nationally (41 deaths), 36 of them in Lee County alone.
  See `outputs/figures/08_ian_impact_layer.png`.

### Data sources added in steps 3-6

- `data/raw/nhc_verification/OFCL_ATL_annual_trk_errors.pdf` / `OFCL_ATL_annual_int_errors.pdf` — NHC's
  published annual average official forecast error tables (nhc.noaa.gov/verification), used as the
  benchmark in the Ian case study.
- `data/raw/nhc_verification/AL092022_Ian_TCR.pdf` — NHC's official Tropical Cyclone Report for Ian,
  used for the impact layer's storm-surge figures.

### Known limitations / natural next steps

- No sea-surface-temperature or ocean-heat-content data in the feature set — the single biggest likely
  improvement for intensity/RI prediction.
- Central pressure excluded from features (missing for ~40% of the historical record pre-satellite-era).
- NHC benchmark is a 2022-season average across all Atlantic storms, not an Ian-specific advisory-by-advisory
  comparison — that would require NHC's ATCF "a-deck" (OFCL) forecast archive, a separate dataset not yet
  pulled in.
- Impact layer uses NHC's surveyed surge heights rather than a full elevation/flood-zone GIS overlay
  (FEMA/USGS DEM data), which would need geopandas/rasterio and larger downloads.
