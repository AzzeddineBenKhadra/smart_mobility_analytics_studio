# 🚦 Smart Mobility Analytics Studio

An interactive Streamlit app for cleaning, exploring, mapping and forecasting transportation data
(road-sensor speeds/volumes and GPS trajectories).

> **Status:** portfolio project. The bundled sample data is **synthetic (simulated)** and exists only to demo
> the app. Findings on it are not real traffic conclusions.

## What it does

| Page | What you get |
|---|---|
| Overview / Quality / Explorer | Auto column detection, data-quality profile, interactive charts |
| Cleaning | Smart cleaning (duplicates, bad coordinates, impossible values, gaps) with undo and a log |
| **Mobility Insights** | Peak hours, congestion index, planning-time index, segment ranking, speed–flow capacity, incident detection, auto-written findings |
| **Trips & OD** | Stops and trips from GPS traces, GPS-glitch removal, origin–destination flows and matrix |
| **Maps** | Points, heatmap, congestion-coloured segments, hourly animation, grid zones, GeoJSON district choropleth |
| **Forecasting** | Lag/rolling/calendar features, HistGradientBoosting / Random Forest / Ridge / LightGBM, 4 baselines, prediction intervals, time-series CV, horizon comparison |
| Report & Export | HTML report (print to PDF), CSV, Excel, Parquet |

## Method notes (what to say in an interview)

- **Congestion index** = 1 − speed / free-flow speed, free-flow = 85th percentile of each road's speed.
- **Planning time index** = free-flow speed ÷ 5th-percentile speed (reliability).
- **Incidents**: speed compared with that road's typical value for the same weekday and hour (robust z-score, or IsolationForest);
  consecutive flags are merged into one incident.
- **Forecast evaluation is leakage-aware**: chronological split, a purge gap equal to the horizon, and every model is
  scored against persistence, seasonal-naive, typical-profile and train-mean baselines on the *same* rows.
  "Skill" is the improvement over the best baseline; it can be negative, and the app says so.
- **Prediction intervals** are empirical and constant-width (from residuals on a held-out calibration slice), not per-point.
- **Stops/trips** depend on GPS sampling rate; thresholds are adjustable.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then click a sample button in the sidebar, or upload CSV / Excel / Parquet / ZIP.

## Test

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

`tests/test_pages_smoke.py` runs every page against a stubbed Streamlit to catch Python errors.
It does **not** check visual layout — always click through the real app before publishing.

## Deploy (Streamlit Community Cloud)

1. Push this folder to a public GitHub repo.
2. At share.streamlit.io choose the repo, branch and `app.py`.
3. Add the link and 3–4 screenshots (put them in `docs/screenshots/`) to this README.

## Project layout

```
app.py            entry point
views/            one module per page
mobility/         logic (no Streamlit): detection, quality, insights, trips, forecasting, geo, mapping, report, dataio
config/field_aliases.json   add your own column-name aliases without touching code
tests/            pytest suite
docs/             case study template, dataset notes
scripts/          make_sample_data.py
```

## Known limitations

- Samples are simulated; the case-study document is a template you must fill with your own findings.
- Very large files: install `pyarrow` (Parquet) and `duckdb` (fast CSV) and prefer Parquet.
- Logs are written to `logs/app.log`.
