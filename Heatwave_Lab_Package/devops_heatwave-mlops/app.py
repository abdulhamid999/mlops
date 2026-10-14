from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("model/heatwave_model.pkl")

FEATURES = ["max_temp", "min_temp", "humidity", "wind_speed", "previous_temp"]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400

    features = [[data[f] for f in FEATURES]]
    prediction = model.predict(features)[0]
    return jsonify({"prediction": "Heatwave" if prediction == 1 else "No Heatwave"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
