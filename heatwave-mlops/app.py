from flask import Flask, request, jsonify
import joblib, json, datetime
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load("model/heatwave_model.pkl")
with open("model/model_meta.json") as f:
    META = json.load(f)
with open("model/normal_by_doy.json") as f:
    NORMAL_BY_DOY = json.load(f)
FEATURE_ORDER = META["feature_order"]


def day_of_year_noleap(d):
    month, day = d.month, d.day
    if month == 2 and day == 29:
        day = 28
    return datetime.date(2001, month, day).timetuple().tm_yday


def build_features(payload):
    required = ["date", "max_temp", "min_temp", "rain", "recent_max_temps"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    try:
        d = datetime.date.fromisoformat(payload["date"])
    except Exception:
        raise ValueError("date must be an ISO date string, e.g. '2024-06-10'")

    try:
        max_temp = float(payload["max_temp"]); min_temp = float(payload["min_temp"]); rain = float(payload["rain"])
    except (TypeError, ValueError):
        raise ValueError("max_temp, min_temp and rain must be numbers")

    if not (-10 <= max_temp <= 55) or not (-10 <= min_temp <= 45):
        raise ValueError("max_temp/min_temp are outside a physically plausible range for Delhi (-10 to 55C)")
    if max_temp < min_temp:
        raise ValueError("max_temp cannot be less than min_temp")
    if rain < 0:
        raise ValueError("rain cannot be negative")

    recent = payload["recent_max_temps"]
    if not isinstance(recent, list) or len(recent) < 3:
        raise ValueError("recent_max_temps must be a list of at least 3 recent daily max temperatures, oldest first, ending with today")
    recent = [float(x) for x in recent]

    warnings = []
    roll3 = float(np.mean(recent[-3:]))
    if len(recent) >= 7:
        roll7 = float(np.mean(recent[-7:]))
    else:
        roll7 = float(np.mean(recent))
        warnings.append(f"Only {len(recent)} days of history supplied; roll7_max computed from {len(recent)} days instead of 7.")

    doy = day_of_year_noleap(d)
    departure = max_temp - NORMAL_BY_DOY[str(doy)]
    condition_day = int(max_temp >= 40 and (departure >= 4.5 or max_temp >= 45))

    features = {"max_temp": max_temp, "min_temp": min_temp, "rain": rain, "departure": departure,
                "condition_day": condition_day, "roll3_max": roll3, "roll7_max": roll7, "month": d.month}
    return features, warnings


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": META["model_name"], "version": META["version"]})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be JSON"}), 400
    try:
        features, warnings = build_features(payload)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    x = pd.DataFrame([[features[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
    pred = int(model.predict(x)[0])
    proba = float(model.predict_proba(x)[0][1])
    response = {
        "date_of_observation": payload["date"], "prediction_for": "next day",
        "prediction": "Heatwave" if pred == 1 else "No Heatwave",
        "heatwave_probability": round(proba, 4),
        "features_used": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in features.items()},
    }
    if warnings:
        response["warnings"] = warnings
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)