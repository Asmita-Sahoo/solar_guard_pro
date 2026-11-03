"""Train the weather -> soiling-loss regression model."""
import os
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ["pm25", "pm10", "humidity", "wind", "rain7", "temp", "days_since_clean"]

df = pd.read_csv("data/weather_soiling.csv")
X, y = df[FEATURES], df["loss_pct"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
model.fit(X_tr, y_tr)

pred = model.predict(X_te)
print(f"MAE : {mean_absolute_error(y_te, pred):.2f} percentage points")
print(f"R^2 : {r2_score(y_te, pred):.3f}")
print("\nFeature importance:")
for name, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda t: -t[1]):
    print(f"  {name:18s} {imp:.3f}")

os.makedirs("models", exist_ok=True)
joblib.dump({"model": model, "features": FEATURES}, "models/weather_model.joblib")
print("\nSaved models/weather_model.joblib")