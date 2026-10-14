"""
mlflow_experiments.py -- runs the 5 tracked configurations from Experiment 2
and registers the selected model. Kept as a standalone script (in addition to
the Experiment 2 notebook) so the MLflow tracking store can be regenerated
independently, e.g. after this project directory is moved or re-cloned.
"""
import pandas as pd
import mlflow, mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import train as pipeline  # reuse the exact same cleaning/labeling/feature code

df = pipeline.load_and_clean("data/heatwave_data.csv")
df, normal = pipeline.build_climatology(df)
df = pipeline.build_imd_label(df)
data, feat_cols = pipeline.build_features(df)

years = data.Date.dt.year
train_mask, test_mask = years <= 2010, years >= 2011
X_train, X_test = data.loc[train_mask, feat_cols], data.loc[test_mask, feat_cols]
y_train, y_test = data.loc[train_mask, "target_tomorrow"], data.loc[test_mask, "target_tomorrow"]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Delhi Heatwave Next-Day Forecast")

configs = [
    {"n_estimators": 50,  "max_depth": 3,  "class_weight": "balanced"},
    {"n_estimators": 100, "max_depth": 5,  "class_weight": "balanced"},
    {"n_estimators": 200, "max_depth": 10, "class_weight": "balanced"},
    {"n_estimators": 200, "max_depth": 6,  "class_weight": "balanced"},
    {"n_estimators": 200, "max_depth": 6,  "class_weight": None},
]

results = []
for cfg in configs:
    with mlflow.start_run(run_name=f"n{cfg['n_estimators']}_d{cfg['max_depth']}_cw{cfg['class_weight']}"):
        model = RandomForestClassifier(random_state=42, **cfg)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        for k, v in cfg.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("false_negatives", fn)
        mlflow.log_metric("false_positives", fp)
        mlflow.sklearn.log_model(model, "model", skops_trusted_types=["sklearn.tree._tree.Tree"])

        run_id = mlflow.active_run().info.run_id
        results.append({**cfg, "accuracy": acc, "precision": prec, "recall": rec,
                         "f1": f1, "fn": fn, "fp": fp, "run_id": run_id})
        print(f"Run {run_id[:8]}  {cfg}  acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}  FN={fn} FP={fp}")

# Select on recall/F1 balance among the class-weighted runs, not top-F1 overall
# (see Experiment 2's report for why the unweighted run is a worse choice
# despite scoring best on accuracy/F1).
runs = mlflow.search_runs(experiment_names=["Delhi Heatwave Next-Day Forecast"])
balanced = runs[runs["params.class_weight"] == "balanced"].sort_values("metrics.f1", ascending=False)
best_run_id = balanced.iloc[0]["run_id"]
print(f"\nSelected run: {best_run_id}  F1: {balanced.iloc[0]['metrics.f1']:.4f}  recall: {balanced.iloc[0]['metrics.recall']}")

model_uri = f"runs:/{best_run_id}/model"
result = mlflow.register_model(model_uri, "delhi-heatwave-nextday-rf")
print(f"Registered: {result.name}  version: {result.version}")
