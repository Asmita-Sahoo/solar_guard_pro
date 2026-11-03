"""SolarGuard - Flask web app: dust/pollution detection + weather-aware loss estimate."""
import io
import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request
from PIL import Image

from features import extract_features

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit

image_model = joblib.load("models/image_model.joblib")
_wx = joblib.load("models/weather_model.joblib")
weather_model, WX_FEATURES = _wx["model"], _wx["features"]

# Typical power loss (%) per class. Rough planning values - calibrate with real data.
CLASS_LOSS = {"clean": 1.0, "light_dust": 8.0, "heavy_dust": 25.0, "bird_dropping": 10.0}


def fetch_weather(lat, lon):
    """Live weather + air quality from Open-Meteo (free, no API key)."""
    w = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
                "daily": "precipitation_sum", "past_days": 7, "forecast_days": 4,
                "timezone": "auto"},
        timeout=8).json()
    a = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={"latitude": lat, "longitude": lon, "current": "pm10,pm2_5"},
        timeout=8).json()
    rain = [p or 0 for p in w["daily"]["precipitation_sum"]]   # 7 past days, today, 3 ahead
    return {
        "temp": w["current"]["temperature_2m"],
        "humidity": w["current"]["relative_humidity_2m"],
        "wind": round(w["current"]["wind_speed_10m"] / 3.6, 2),  # km/h -> m/s
        "pm25": a["current"]["pm2_5"],
        "pm10": a["current"]["pm10"],
        "rain7": round(sum(rain[:7]), 1),
        "rain_forecast": round(sum(rain[7:]), 1),                # today + next 3 days
    }


def recommend(loss, top_class, rain_forecast):
    if top_class == "bird_dropping" and loss < 15:
        return "Spot-clean the bird droppings (hot spots can damage cells)."
    if loss < 5:
        return "Panels are in good condition. No cleaning needed."
    if loss < 12:
        if rain_forecast >= 5:
            return f"Moderate soiling, but {rain_forecast} mm of rain is forecast. Wait and re-check after the rain."
        return "Moderate soiling. Schedule cleaning within a week."
    if loss < 20:
        return "High soiling. Clean within 2-3 days."
    return "Severe soiling. Clean immediately - large energy loss."


def analyze(img, wx, days, capacity_kw):
    probs = image_model.predict_proba(extract_features(img).reshape(1, -1))[0]
    classes = list(image_model.classes_)
    top = classes[int(np.argmax(probs))]
    image_loss = float(sum(p * CLASS_LOSS.get(c, 5.0) for p, c in zip(probs, classes)))

    row = pd.DataFrame([[wx["pm25"], wx["pm10"], wx["humidity"], wx["wind"],
                         wx["rain7"], wx["temp"], days]], columns=WX_FEATURES)
    weather_loss = float(weather_model.predict(row)[0])

    # Heuristic fusion: the photo is direct evidence, the weather model is context.
    final = round(0.6 * image_loss + 0.4 * weather_loss, 1)
    return {
        "dust_class": top,
        "confidence": round(float(max(probs)) * 100, 1),
        "class_probs": {c: round(float(p) * 100, 1) for c, p in zip(classes, probs)},
        "image_loss_pct": round(image_loss, 1),
        "weather_loss_pct": round(weather_loss, 1),
        "final_loss_pct": final,
        "power_lost_kw": round(capacity_kw * final / 100, 2),
        "weather": wx,
        "recommendation": recommend(final, top, wx.get("rain_forecast", 0)),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    f = request.files.get("image")
    if not f:
        return jsonify(error="Please upload a solar panel image."), 400
    try:
        img = Image.open(io.BytesIO(f.read()))
        img.load()
    except Exception:
        return jsonify(error="Could not read that file as an image."), 400

    try:
        days = int(request.form.get("days", 14))
        capacity = float(request.form.get("capacity", 5))
        if request.form.get("mode", "auto") == "auto":
            wx = fetch_weather(float(request.form["lat"]), float(request.form["lon"]))
        else:
            g = lambda k: float(request.form[k])
            wx = {"pm25": g("pm25"), "pm10": g("pm10"), "humidity": g("humidity"),
                  "wind": g("wind"), "rain7": g("rain7"), "temp": g("temp"),
                  "rain_forecast": float(request.form.get("rain_forecast", 0))}
    except (requests.RequestException, KeyError, ValueError, TypeError):
        return jsonify(error="Weather data unavailable. Check inputs or switch to manual mode."), 502

    return jsonify(analyze(img, wx, days, capacity))


if __name__ == "__main__":
    app.run(debug=True)