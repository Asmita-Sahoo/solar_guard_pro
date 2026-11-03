"""Camera access + two helper commands.

    python camera.py preview                      save preview.jpg (ROI box drawn) to check the view
    python camera.py collect clean 40 5           save 40 cropped frames, 5 s apart, into dataset/clean
"""
import glob
import io
import os
import random
import sys
import time

from PIL import Image, ImageDraw

import config

IMG_EXT = (".png", ".jpg", ".jpeg")


def capture_frame():
    """Return one frame from the configured source as an RGB PIL image."""
    src = config.CAMERA_SOURCE

    # Test mode: random image from a folder (no camera needed)
    if isinstance(src, str) and src.startswith("folder:"):
        files = [p for p in glob.glob(os.path.join(src[7:], "**", "*"), recursive=True)
                 if p.lower().endswith(IMG_EXT)]
        if not files:
            raise RuntimeError("No images found in the test folder")
        return Image.open(random.choice(files)).convert("RGB")

    # Snapshot URL (phone IP-camera apps, ESP32-CAM /capture ...)
    if isinstance(src, str) and src.lower().startswith("http") \
            and src.lower().split("?")[0].endswith((".jpg", ".jpeg", "/capture")):
        import requests
        r = requests.get(src, timeout=10)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")

    # Webcam / USB camera / RTSP stream
    import cv2
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if isinstance(src, int) and os.name == "nt" \
        else cv2.VideoCapture(src)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {src}")
        frame = None
        for _ in range(5):                    # let auto-exposure settle
            ok, f = cap.read()
            if ok:
                frame = f
        if frame is None:
            raise RuntimeError("Camera opened but returned no frame")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()


def crop_roi(img):
    """Keep only the solar-panel region defined in config.PANEL_ROI."""
    if not config.PANEL_ROI:
        return img
    w, h = img.size
    x1, y1, x2, y2 = config.PANEL_ROI
    return img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "preview"

    if cmd == "preview":
        img = capture_frame()
        if config.PANEL_ROI:
            w, h = img.size
            x1, y1, x2, y2 = config.PANEL_ROI
            ImageDraw.Draw(img).rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=(255, 0, 0), width=4)
        img.save("preview.jpg")
        print("Saved preview.jpg - open it and check the panel is inside the red box.")

    elif cmd == "collect" and len(sys.argv) >= 4:
        label, n = sys.argv[2], int(sys.argv[3])
        gap = float(sys.argv[4]) if len(sys.argv) > 4 else 5
        os.makedirs(f"dataset/{label}", exist_ok=True)
        for i in range(n):
            name = f"dataset/{label}/real_{time.strftime('%Y%m%d_%H%M%S')}_{i:03d}.jpg"
            crop_roi(capture_frame()).save(name, quality=92)
            print(f"[{i + 1}/{n}] saved {name}")
            time.sleep(gap)
    else:
        print(__doc__)