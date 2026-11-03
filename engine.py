"""One monitoring cycle: quality check -> classify -> weather -> fuse -> store -> alert."""
import os
import threading
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

import config
import db
from app import CLASS_LOSS, WX_FEATURES, fetch_weather, image_model, recommend, weather_model
from camera import crop_roi
from features import extract_features

_lock = threading.Lock()      # only one capture/analysis at a time
_last_weather = None


def get_weather():
    """Live weather -> last good weather -> defaults (so the system never stops)."""
    global _last_weather
    try:
        wx = fetch_weather(config.LATITUDE, config.LONGITUDE)
        _last_weather = wx
        return wx, "live"
    except Exception:
        if _last_weather:
            return _last_weather, "cached"
        return dict(config.DEFAULT_WEATHER), "default"


def quality_problem(img):
    """Return a reason string if the frame is unusable (night, glare), else None."""
    b = float(np.asarray(img.convert("L"), dtype=np.float32).mean() / 255)
    if b < config.MIN_BRIGHTNESS:
        return f"too dark (brightness {b:.2f})"
    if b > config.MAX_BRIGHTNESS:
        return f"overexposed (brightness {b:.2f})"
    return None


def send_alert(text):
    print("ALERT:", text)
    with open("alerts.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M}  {text}\n")
    if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
        try:
            requests.post(f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
                          data={"chat_id": config.TELEGRAM_CHAT_ID, "text": text}, timeout=10)
        except requests.RequestException as e:
            print("Telegram alert failed:", e)


def _cooldown_over():
    last = db.last_alert_time()
    return last is None or datetime.now() - last > timedelta(hours=config.ALERT_COOLDOWN_HOURS)


def run_cycle(img):
    """Analyse one camera frame, save everything, return the result dict."""
    with _lock:
        os.makedirs(config.CAPTURE_DIR, exist_ok=True)
        now = datetime.now()
        name = now.strftime("%Y%m%d_%H%M%S") + ".jpg"
        img = img.convert("RGB")
        img.save(os.path.join(config.CAPTURE_DIR, name), quality=90)
        res = {"ts": now.isoformat(timespec="seconds"), "image": name}

        panel = crop_roi(img)
        problem = quality_problem(panel)
        if problem:
            res["status"] = "skipped: " + problem
            db.add_reading(res)
            return res

        probs = image_model.predict_proba(extract_features(panel).reshape(1, -1))[0]
        classes = list(image_model.classes_)
        top = classes[int(np.argmax(probs))]
        conf = float(max(probs)) * 100
        image_loss = float(sum(p * CLASS_LOSS.get(c, 5.0) for p, c in zip(probs, classes)))

        wx, wsrc = get_weather()
        days = db.days_since_cleaning()
        row = pd.DataFrame([[wx["pm25"], wx["pm10"], wx["humidity"], wx["wind"],
                             wx["rain7"], wx["temp"], days]], columns=WX_FEATURES)
        weather_loss = float(weather_model.predict(row)[0])

        final = round(0.6 * image_loss + 0.4 * weather_loss, 1)
        kw = config.CAPACITY_KW * final / 100
        kwh = kw * config.PEAK_SUN_HOURS
        res.update({
            "status": "ok" if conf >= config.MIN_CONFIDENCE else "low confidence - verify manually",
            "dust_class": top, "confidence": round(conf, 1),
            "image_loss": round(image_loss, 1), "weather_loss": round(weather_loss, 1),
            "final_loss": final, "power_lost_kw": round(kw, 2),
            "daily_kwh_lost": round(kwh, 1), "daily_cost_lost": round(kwh * config.TARIFF_PER_KWH, 1),
            "recommendation": recommend(final, top, wx.get("rain_forecast", 0)),
            "weather": wx, "weather_source": wsrc, "alerted": 0,
        })

        if final >= config.ALERT_LOSS_PCT and _cooldown_over():
            send_alert(f"SolarGuard: {top.replace('_', ' ')}, estimated loss {final}% "
                       f"(~Rs {res['daily_cost_lost']}/day). {res['recommendation']}")
            res["alerted"] = 1

        db.add_reading(res)
        return res