"""Promote the MLflow-registered model to a standalone joblib file for the
prediction service -- this decouples the serving app from needing a live
MLflow tracking connection at request time."""
import mlflow, mlflow.sklearn, joblib, json

mlflow.set_tracking_uri('sqlite:///mlflow.db')
model = mlflow.sklearn.load_model('models:/delhi-heatwave-nextday-rf/1')
joblib.dump(model, 'model/heatwave_model.pkl')

meta = {
    "model_name": "delhi-heatwave-nextday-rf",
    "version": 1,
    "algorithm": "RandomForestClassifier",
    "params": model.get_params(),
    "feature_order": ["max_temp","min_temp","rain","departure","condition_day",
                       "roll3_max","roll7_max","month"],
}
with open('model/model_meta.json', 'w') as f:
    json.dump(meta, f, indent=2, default=str)
print("Saved model/heatwave_model.pkl and model/model_meta.json")
