# heatwave-mlops

Heatwave prediction service — ML → MLflow → MLOps → DevOps.

## Run locally

```bash
pip install -r requirements.txt
python train.py               # trains model/heatwave_model.pkl
python app.py                 # starts the API on :8000
```

## Test the API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"max_temp":45,"min_temp":31,"humidity":50,"wind_speed":7,"previous_temp":44}'
```

## Docker

```bash
docker build -t heatwave-api:1.0 .
docker run -p 8000:8000 heatwave-api:1.0
```

## CI

`.github/workflows/ci.yml` runs on every push to `main`: installs
dependencies, compiles the app, and re-trains the model as a smoke test.
