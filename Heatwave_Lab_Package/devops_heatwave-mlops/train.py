"""
train.py — retrains the heatwave model from data/heatwave_data.csv and
saves it to model/heatwave_model.pkl. Run this once before building the
Docker image, or wire it into the CI workflow as a training/validation step.
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_PATH = "data/heatwave_data.csv"
MODEL_PATH = "model/heatwave_model.pkl"

FEATURES = ["max_temp", "min_temp", "humidity", "wind_speed", "previous_temp"]
TARGET = "heatwave"


def main():
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Validation accuracy: {accuracy:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
