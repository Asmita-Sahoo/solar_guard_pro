"""SQLite storage: every reading and every cleaning event."""
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta

import config


def _conn():
    c = sqlite3.connect(config.DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def _now():
    return datetime.now().isoformat(timespec="seconds")


def init():
    with closing(_conn()) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS readings(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, image TEXT, status TEXT,
            dust_class TEXT, confidence REAL, image_loss REAL, weather_loss REAL,
            final_loss REAL, power_lost_kw REAL, daily_kwh_lost REAL, daily_cost_lost REAL,
            recommendation TEXT, weather TEXT, weather_source TEXT, alerted INTEGER DEFAULT 0)""")
        c.execute("CREATE TABLE IF NOT EXISTS cleanings(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT)")
        if c.execute("SELECT COUNT(*) FROM cleanings").fetchone()[0] == 0:
            first = datetime.now() - timedelta(days=config.INITIAL_DAYS_SINCE_CLEAN)
            c.execute("INSERT INTO cleanings(ts) VALUES (?)", (first.isoformat(timespec="seconds"),))
        c.commit()


def days_since_cleaning():
    with closing(_conn()) as c:
        ts = c.execute("SELECT ts FROM cleanings ORDER BY id DESC LIMIT 1").fetchone()[0]
    return max(0, (datetime.now() - datetime.fromisoformat(ts)).days)


def mark_cleaned():
    with closing(_conn()) as c:
        c.execute("INSERT INTO cleanings(ts) VALUES (?)", (_now(),))
        c.commit()


def add_reading(r):
    with closing(_conn()) as c:
        c.execute("""INSERT INTO readings(ts, image, status, dust_class, confidence, image_loss,
                     weather_loss, final_loss, power_lost_kw, daily_kwh_lost, daily_cost_lost,
                     recommendation, weather, weather_source, alerted)
                     VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (r["ts"], r["image"], r["status"], r.get("dust_class"), r.get("confidence"),
                   r.get("image_loss"), r.get("weather_loss"), r.get("final_loss"),
                   r.get("power_lost_kw"), r.get("daily_kwh_lost"), r.get("daily_cost_lost"),
                   r.get("recommendation"), json.dumps(r.get("weather")), r.get("weather_source"),
                   r.get("alerted", 0)))
        c.commit()


def history(limit=60):
    """Newest first."""
    with closing(_conn()) as c:
        rows = c.execute("SELECT * FROM readings ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        d["weather"] = json.loads(d["weather"]) if d["weather"] else None
        out.append(d)
    return out


def last_alert_time():
    with closing(_conn()) as c:
        row = c.execute("SELECT ts FROM readings WHERE alerted=1 ORDER BY id DESC LIMIT 1").fetchone()
    return datetime.fromisoformat(row[0]) if row else None