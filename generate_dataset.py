"""Create a synthetic starter dataset so the whole project runs end-to-end."""
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

CLASSES = ["clean", "light_dust", "heavy_dust", "bird_dropping"]
WEATHERS = ["sunny", "cloudy", "foggy", "hazy"]
SIZE = 224


def smooth_noise(rng, cells=12):
    low = rng.random((cells, cells)).astype(np.float32)
    img = Image.fromarray((low * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def base_panel(rng):
    img = np.zeros((SIZE, SIZE, 3), np.float32)
    img[:] = rng.uniform([15, 30, 90], [35, 60, 140])
    img += np.linspace(0, rng.uniform(5, 25), SIZE)[None, :, None]
    step = SIZE // 6
    for i in range(0, SIZE, step):
        img[i:i + 2, :, :] = [170, 175, 185]
        img[:, i:i + 2, :] = [170, 175, 185]
    return img


def add_dust(img, rng, alpha_range):
    a = rng.uniform(*alpha_range)
    mask = np.clip(a * (0.6 + 0.8 * smooth_noise(rng)), 0, 0.95)[..., None]
    color = rng.uniform([150, 125, 95], [200, 175, 140])
    return img * (1 - mask) + color * mask


def add_droppings(img, rng):
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    for _ in range(int(rng.integers(2, 6))):
        x, y = (int(v) for v in rng.integers(20, SIZE - 40, 2))
        w, h = (int(v) for v in rng.integers(10, 35, 2))
        c = tuple(int(v) for v in rng.uniform(210, 245, 3))
        d.ellipse([x, y, x + w, y + h], fill=c)
        d.ellipse([x + w // 3, y + h // 3, x + w // 2, y + h // 2], fill=(120, 110, 90))
    return np.asarray(pil, dtype=np.float32)


def apply_weather(img, rng, weather):
    if weather == "sunny":
        img = img * rng.uniform(1.05, 1.25)
    elif weather == "cloudy":
        img = img * rng.uniform(0.65, 0.85)
    elif weather == "foggy":
        f = rng.uniform(0.25, 0.5)
        img = img * (1 - f) + np.array([200, 200, 205]) * f
    elif weather == "hazy":
        f = rng.uniform(0.1, 0.3)
        img = img * (1 - f) + np.array([190, 170, 140]) * f
    img = img + rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_panel(kind, rng, weather):
    img = base_panel(rng)
    if kind == "light_dust":
        img = add_dust(img, rng, (0.12, 0.30))
    elif kind == "heavy_dust":
        img = add_dust(img, rng, (0.45, 0.80))
    elif kind == "bird_dropping":
        img = add_droppings(add_dust(img, rng, (0.0, 0.05)), rng)
    return Image.fromarray(apply_weather(img, rng, weather))


def make_images(per_class=150, seed=42):
    rng = np.random.default_rng(seed)
    for kind in CLASSES:
        os.makedirs(f"dataset/{kind}", exist_ok=True)
        for i in range(per_class):
            weather = WEATHERS[i % len(WEATHERS)]
            make_panel(kind, rng, weather).save(f"dataset/{kind}/{kind}_{i:04d}.png")
    print(f"Images: {per_class * len(CLASSES)} written to dataset/")


def make_weather_csv(n=4000, seed=7):
    """Physics-inspired synthetic data. Replace with real plant data later."""
    rng = np.random.default_rng(seed)
    pm25 = rng.gamma(3, 15, n).clip(2, 300)
    pm10 = (pm25 * rng.uniform(1.2, 2.2, n)).clip(5, 500)
    humidity = rng.uniform(15, 100, n)
    wind = rng.gamma(2, 2, n).clip(0, 20)                      # m/s
    rain7 = rng.exponential(8, n) * (rng.random(n) < 0.45)    # mm in last 7 days
    temp = rng.uniform(5, 45, n)
    days = rng.integers(0, 60, n)                              # days since last cleaning

    rate = 0.03 + 0.0010 * pm10 + 0.0018 * pm25                # % loss per day
    rate = rate * (1 + 0.4 * (humidity > 70) + 0.02 * np.clip(wind - 3, 0, None))
    wash = np.clip(rain7 / 12, 0, 1) * 0.85                    # rain removes dust
    loss = (rate * days * (1 - wash) + rng.normal(0, 0.8, n)).clip(0, 45)

    df = pd.DataFrame({"pm25": pm25, "pm10": pm10, "humidity": humidity, "wind": wind,
                       "rain7": rain7, "temp": temp, "days_since_clean": days,
                       "loss_pct": loss}).round(2)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/weather_soiling.csv", index=False)
    print(f"Weather table: {n} rows written to data/weather_soiling.csv")


if __name__ == "__main__":
    per_class = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    make_images(per_class)
    make_weather_csv()