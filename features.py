"""Image feature extraction shared by training and the web app."""
import numpy as np
from PIL import Image


def extract_features(img: Image.Image) -> np.ndarray:
    """Turn a solar-panel photo into a 24-number feature vector."""
    img = img.convert("RGB").resize((128, 128))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)

    feats = []
    feats += list(arr.mean(axis=(0, 1)))            # mean R, G, B
    feats += list(arr.std(axis=(0, 1)))             # std  R, G, B
    feats += [gray.mean(), gray.std()]              # brightness, contrast

    mx, mn = arr.max(axis=2), arr.min(axis=2)
    sat = (mx - mn) / (mx + 1e-6)
    feats += [sat.mean(), sat.std()]                # colour saturation

    feats += [np.abs(np.diff(gray, axis=1)).mean(),  # edge strength (x)
              np.abs(np.diff(gray, axis=0)).mean()]  # edge strength (y)

    blocks = gray.reshape(8, 16, 8, 16).std(axis=(1, 3))   # local texture
    feats += [blocks.mean(), blocks.std(), blocks.max()]

    hist, _ = np.histogram(gray, bins=8, range=(0, 1), density=True)
    feats += list(hist)                              # brightness histogram

    feats += [(arr[:, :, 2] - arr[:, :, 0]).mean()]  # "blueness"
    return np.array(feats, dtype=np.float32)