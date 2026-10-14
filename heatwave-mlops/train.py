"""
train.py -- reproduces the full Delhi heat-wave forecasting pipeline from raw
data to a saved model artifact. This is the single source of truth for how
model/heatwave_model.pkl, model/model_meta.json and model/normal_by_doy.json
are produced -- CI runs this so the served model can never silently drift
from a documented, reproducible training process.

Usage: python train.py --data data/heatwave_data.csv
"""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


def load_and_clean(path):
    raw = pd.read_csv(path, dtype=str)

    def parse_date(s):
        if s.isdigit() and len(s) == 5:  # Excel serial-number date artifact
            return pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(s))
        return pd.to_datetime(s, format="%d-%m-%Y")

    raw["Date"] = raw["Date"].apply(parse_date)
    raw["Rain"] = raw["Rain"].replace("Tr", "0.0")  # IMD "Trace" rainfall
    for col in ["Rain", "Temp Max", "Temp Min"]:
        raw[col] = raw[col].replace({"----": np.nan, "-----": np.nan})
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    df = raw.sort_values("Date").reset_index(drop=True)
    df = df.rename(columns={"Temp Max": "max_temp", "Temp Min": "min_temp", "Rain": "rain"})
    df = df.dropna(subset=["max_temp", "min_temp"]).reset_index(drop=True)
    return df


def build_climatology(df):
    month = df.Date.dt.month
    day = df.Date.dt.day.where(~((month == 2) & (df.Date.dt.day == 29)), 28)
    ref_date = pd.to_datetime({"year": 2001, "month": month, "day": day})
    df["doy"] = ref_date.dt.dayofyear

    base = df[(df.Date.dt.year >= 1991) & (df.Date.dt.year <= 2020)]
    raw_normal = base.groupby("doy")["max_temp"].mean().reindex(range(1, 366))
    window = 7
    padded = pd.concat([raw_normal, raw_normal, raw_normal])
    padded.index = range(-365, 365 * 2)
    normal = pd.Series({d: padded.loc[d - window:d + window].mean() for d in range(1, 366)})
    df["normal_max_temp"] = df["doy"].map(normal)
    df["departure"] = df["max_temp"] - df["normal_max_temp"]
    return df, normal


def build_imd_label(df):
    qualifies = df["max_temp"] >= 40
    condition_day = qualifies & ((df["departure"] >= 4.5) | (df["max_temp"] >= 45))
    df["condition_day"] = condition_day.astype(int)
    grp = (df["condition_day"] != df["condition_day"].shift()).cumsum()
    run_len = df.groupby(grp)["condition_day"].transform("size")
    df["official_heatwave_day"] = ((df["condition_day"] == 1) & (run_len >= 2)).astype(int)
    return df


def build_features(df):
    df["rain"] = df["rain"].fillna(0)
    df["roll3_max"] = df["max_temp"].rolling(3, min_periods=3).mean()
    df["roll7_max"] = df["max_temp"].rolling(7, min_periods=7).mean()
    df["month"] = df.Date.dt.month
    df["target_tomorrow"] = df["official_heatwave_day"].shift(-1)

    feat_cols = ["max_temp", "min_temp", "rain", "departure", "condition_day",
                 "roll3_max", "roll7_max", "month"]
    data = df.dropna(subset=feat_cols + ["target_tomorrow"]).reset_index(drop=True)
    data["target_tomorrow"] = data["target_tomorrow"].astype(int)
    return data, feat_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/heatwave_data.csv")
    parser.add_argument("--out-dir", default="model")
    args = parser.parse_args()

    df = load_and_clean(args.data)
    df, normal = build_climatology(df)
    df = build_imd_label(df)
    data, feat_cols = build_features(df)

    years = data.Date.dt.year
    X_train = data.loc[years <= 2010, feat_cols]
    y_train = data.loc[years <= 2010, "target_tomorrow"]

    # Configuration selected in Experiment 2 by comparing tracked MLflow runs:
    # best precision/F1 among runs achieving recall >= 0.88.
    model = RandomForestClassifier(n_estimators=200, max_depth=10,
                                    class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.out_dir, "heatwave_model.pkl"))

    meta = {
        "model_name": "delhi-heatwave-nextday-rf",
        "version": 1,
        "algorithm": "RandomForestClassifier",
        "params": {k: str(v) for k, v in model.get_params().items()},
        "feature_order": feat_cols,
        "trained_on_rows": int(len(X_train)),
        "trained_through_year": 2010,
    }
    with open(os.path.join(args.out_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    normal_by_doy = {str(int(k)): round(float(v), 3) for k, v in normal.to_dict().items()}
    with open(os.path.join(args.out_dir, "normal_by_doy.json"), "w") as f:
        json.dump(normal_by_doy, f)

    print(f"Trained on {len(X_train)} rows (1951-2010). "
          f"Saved model, metadata and climatology to {args.out_dir}/")


if __name__ == "__main__":
    main()
