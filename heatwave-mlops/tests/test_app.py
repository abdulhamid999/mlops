"""
Automated tests for the heat-wave prediction service, run in CI on every push.
Uses Flask's test client, so no live server or Docker is needed to run these --
exactly what a GitHub Actions runner will do.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app  # noqa: E402


def client():
    app.testing = True
    return app.test_client()


def test_health():
    resp = client().get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["model"] == "delhi-heatwave-nextday-rf"


def test_predict_hot_day_forecasts_heatwave():
    resp = client().post("/predict", json={
        "date": "2024-06-09", "max_temp": 44.8, "min_temp": 29.5, "rain": 0.0,
        "recent_max_temps": [41.2, 42.5, 43.0, 43.8, 44.1, 44.5, 44.8],
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["prediction"] == "Heatwave"
    assert body["heatwave_probability"] > 0.5


def test_predict_mild_winter_day_forecasts_no_heatwave():
    resp = client().post("/predict", json={
        "date": "2024-01-15", "max_temp": 19.5, "min_temp": 7.2, "rain": 0.0,
        "recent_max_temps": [18.5, 19.0, 19.2, 19.0, 18.8, 19.1, 19.5],
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["prediction"] == "No Heatwave"
    assert body["heatwave_probability"] < 0.5


def test_predict_missing_fields_returns_400():
    resp = client().post("/predict", json={"date": "2024-06-09", "max_temp": 44.8})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_predict_implausible_temperature_returns_400():
    resp = client().post("/predict", json={
        "date": "2024-06-09", "max_temp": 90, "min_temp": 29.5, "rain": 0.0,
        "recent_max_temps": [41.2, 42.5, 43.0],
    })
    assert resp.status_code == 400


def test_predict_max_temp_less_than_min_temp_returns_400():
    resp = client().post("/predict", json={
        "date": "2024-06-09", "max_temp": 20, "min_temp": 29.5, "rain": 0.0,
        "recent_max_temps": [41.2, 42.5, 43.0],
    })
    assert resp.status_code == 400


def test_predict_non_json_body_returns_400():
    resp = client().post("/predict", data="not json", content_type="text/plain")
    assert resp.status_code == 400


def test_predict_short_history_still_succeeds_with_warning():
    resp = client().post("/predict", json={
        "date": "2024-06-09", "max_temp": 44.8, "min_temp": 29.5, "rain": 0.0,
        "recent_max_temps": [43.5, 44.1, 44.8],
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert "warnings" in body


def test_predict_too_short_history_returns_400():
    resp = client().post("/predict", json={
        "date": "2024-06-09", "max_temp": 44.8, "min_temp": 29.5, "rain": 0.0,
        "recent_max_temps": [44.1, 44.8],  # only 2 days, minimum is 3
    })
    assert resp.status_code == 400
