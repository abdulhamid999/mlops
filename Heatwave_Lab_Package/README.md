# Heatwave Prediction Lab — Completed Package

## Contents
- `notebook/Heatwave_Prediction_MLOps.ipynb` — the full Colab notebook (all 4 experiments), clean cells ready to re-run.
- `notebook/Heatwave_Prediction_MLOps_executed.ipynb` — the same notebook already executed, with real outputs/plots inline (open this to see results without re-running).
- `writeup/Heatwave_Lab_Writeups.docx` (+ matching `.pdf`) — the lab writeup for all 4 experiments: Aim, Procedure, Output tables, embedded charts/screenshots, Result, plus Kubernetes note, Conclusion, and Viva questions.
- `devops_heatwave-mlops/` — the standalone DevOps project (Experiment 4): `app.py`, `train.py`, `requirements.txt`, `Dockerfile`, `README.md`, `.github/workflows/ci.yml`, plus `data/` and the trained `model/heatwave_model.pkl`.
- `data/heatwave_teaching.csv` — the manual's original 20-row dataset (Experiment 1).
- `data/heatwave_timeseries.csv` — the larger 1,095-row (3-year) synthetic dataset used from Experiment 2 onward.
- `images/` — all charts/screenshots used in the writeup, generated from real runs.

## Important note on the Experiment 2-4 dataset
This was built in a sandboxed environment with no access to general dataset
websites (Kaggle, IMD, etc.), so `heatwave_timeseries.csv` is a realistic
synthetic 3-year daily dataset (seasonal temp/humidity/wind curves + an
IMD-style heatwave rule) with the *same column names* as the manual's
dataset. To use a real dataset instead: download one with columns
`max_temp, min_temp, humidity, wind_speed, previous_temp, heatwave` (or
rename yours to match), drop it in as `data/heatwave_timeseries.csv`
(and `devops_heatwave-mlops/data/heatwave_data.csv`), and re-run — no
other code changes needed.

## Docker / GitHub Actions
`docker build` / `docker run` were not executed in this sandbox (no Docker
daemon available here). Run these yourself in `devops_heatwave-mlops/`:
```bash
docker build -t heatwave-api:1.0 .
docker run -p 8000:8000 heatwave-api:1.0
```
Push the folder to a GitHub repo named `heatwave-mlops` to trigger the CI
workflow in `.github/workflows/ci.yml`.
