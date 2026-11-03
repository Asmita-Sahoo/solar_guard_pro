"""SolarGuard Pro - starts the scheduler (camera capture at fixed times) and the dashboard.

    python run_system.py        then open http://127.0.0.1:5001
"""
import threading
import time
from datetime import datetime, timedelta

from flask import Flask, jsonify, render_template, send_from_directory

import config
import db
from camera import capture_frame
from engine import run_cycle

web = Flask(__name__)
state = {"next_run": None, "last_error": None}


def next_run_after(now):
    """When should the next capture happen?"""
    if config.INTERVAL_MINUTES:
        t = now + timedelta(minutes=config.INTERVAL_MINUTES)
        if t.hour < config.DAYLIGHT_START:
            t = t.replace(hour=config.DAYLIGHT_START, minute=0, second=0, microsecond=0)
        elif t.hour >= config.DAYLIGHT_END:
            t = (t + timedelta(days=1)).replace(hour=config.DAYLIGHT_START, minute=0,
                                                 second=0, microsecond=0)
        return t
    candidates = []
    for day in (0, 1):
        for hhmm in config.CAPTURE_TIMES:
            h, m = map(int, hhmm.split(":"))
            t = (now + timedelta(days=day)).replace(hour=h, minute=m, second=0, microsecond=0)
            if t > now:
                candidates.append(t)
    return min(candidates)


def capture_and_analyze():
    try:
        res = run_cycle(capture_frame())
        state["last_error"] = None
        print(f"[{datetime.now():%H:%M:%S}] {res['status']}  "
              f"{res.get('dust_class', '')}  loss={res.get('final_loss', '-')}%")
        return res
    except Exception as e:
        state["last_error"] = f"{datetime.now():%d %b %H:%M} - {e}"
        print("Capture failed:", e)
        return None


def scheduler():
    if config.RUN_ON_START:
        capture_and_analyze()
    state["next_run"] = next_run_after(datetime.now())
    while True:
        if datetime.now() >= state["next_run"]:
            capture_and_analyze()
            state["next_run"] = next_run_after(datetime.now())
        time.sleep(15)


@web.route("/")
def index():
    return render_template("dashboard.html")


@web.route("/api/status")
def api_status():
    nr = state["next_run"]
    return jsonify(history=db.history(60), days_since_cleaning=db.days_since_cleaning(),
                   next_run=nr.strftime("%d %b %H:%M") if nr else None,
                   last_error=state["last_error"], alert_pct=config.ALERT_LOSS_PCT,
                   capacity_kw=config.CAPACITY_KW)


@web.route("/api/capture-now", methods=["POST"])
def api_capture_now():
    res = capture_and_analyze()
    if res is None:
        return jsonify(error=state["last_error"]), 500
    return jsonify(res)


@web.route("/api/mark-cleaned", methods=["POST"])
def api_mark_cleaned():
    db.mark_cleaned()
    return jsonify(ok=True, days_since_cleaning=db.days_since_cleaning())


@web.route("/captures/<path:name>")
def captures(name):
    return send_from_directory(config.CAPTURE_DIR, name)


if __name__ == "__main__":
    db.init()
    threading.Thread(target=scheduler, daemon=True).start()
    fixed = f"every {config.INTERVAL_MINUTES} min" if config.INTERVAL_MINUTES \
        else "at " + ", ".join(config.CAPTURE_TIMES)
    print(f"SolarGuard Pro running. Captures {fixed}. Dashboard: http://127.0.0.1:5001")
    web.run(host="127.0.0.1", port=5001, debug=False)