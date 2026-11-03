"""Train the dust / bird-dropping classifier (Random Forest on image features)."""
import os
import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from features import extract_features

DATA = "dataset"
classes = sorted(d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d)))
print("Classes:", classes)

X, y = [], []
for c in classes:
    folder = os.path.join(DATA, c)
    for f in os.listdir(folder):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            X.append(extract_features(Image.open(os.path.join(folder, f))))
            y.append(c)
X, y = np.array(X), np.array(y)
print("Samples:", len(X))

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_tr, y_tr)

pred = model.predict(X_te)
print(f"\nTest accuracy: {accuracy_score(y_te, pred):.3f}\n")
print(classification_report(y_te, pred))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/image_model.joblib")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_te, pred, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), model.classes_, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), model.classes_)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Image model")
    fig.tight_layout(); fig.savefig("models/confusion_matrix.png", dpi=120)
    print("Saved models/confusion_matrix.png")
except ImportError:
    pass
print("Saved models/image_model.joblib")