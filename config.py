"""All settings in one place. Edit only this file."""

# ---------------- Camera ----------------
# 0                                   -> laptop / USB webcam
# "http://192.168.1.50:8080/shot.jpg" -> phone or IP camera snapshot URL
# "rtsp://user:pass@192.168.1.60:554/stream1" -> CCTV / IP camera stream
# "folder:dataset"                    -> TEST MODE: picks random images, no camera needed
CAMERA_SOURCE = "folder:dataset"

# Part of the frame that contains ONLY the solar panel, as fractions (x1, y1, x2, y2).
# None = use the whole frame.  Example: (0.2, 0.1, 0.8, 0.9)
PANEL_ROI = None

# ---------------- Schedule ----------------
CAPTURE_TIMES = ["09:00", "12:00", "15:00"]   # fixed daily capture times (24h, local time)
INTERVAL_MINUTES = None                        # e.g. 60 = every hour (overrides CAPTURE_TIMES)
DAYLIGHT_START, DAYLIGHT_END = 7, 18           # interval mode captures only between these hours
RUN_ON_START = True                            # take one reading as soon as the system starts

# ---------------- Site ----------------
LATITUDE, LONGITUDE = 20.30, 85.82
CAPACITY_KW = 5.0
INITIAL_DAYS_SINCE_CLEAN = 14      # used only the first time; later use "Mark cleaned" button
PEAK_SUN_HOURS = 5.0               # assumption for daily energy estimate
TARIFF_PER_KWH = 7.0               # Rs per kWh (assumption, change to your tariff)

# Used when live weather cannot be fetched (offline / API down)
DEFAULT_WEATHER = {"pm25": 60, "pm10": 110, "humidity": 55, "wind": 3.0,
                   "rain7": 0, "temp": 32, "rain_forecast": 0}

# ---------------- Quality checks ----------------
MIN_BRIGHTNESS, MAX_BRIGHTNESS = 0.15, 0.95   # skip night / overexposed frames
MIN_CONFIDENCE = 50.0                         # below this %, flag "verify manually"

# ---------------- Alerts ----------------
ALERT_LOSS_PCT = 15.0              # alert when estimated loss reaches this %
ALERT_COOLDOWN_HOURS = 24          # do not repeat the alert more often than this
TELEGRAM_BOT_TOKEN = ""            # optional: phone alerts via a Telegram bot
TELEGRAM_CHAT_ID = ""

# ---------------- Files ----------------
CAPTURE_DIR = "captures"
DB_PATH = "solarguard.db"