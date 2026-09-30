import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    BASE_DIR
    / "artifacts"
    / "data"
    / "processed"
)


# ============================================================
# LOAD PROCESSED DATA
# ============================================================

X_train = np.load(
    PROCESSED_DIR / "X_train.npy"
)

X_test = np.load(
    PROCESSED_DIR / "X_test.npy"
)

y_train = np.load(
    PROCESSED_DIR / "y_train.npy"
)

y_test = np.load(
    PROCESSED_DIR / "y_test.npy"
)

print("\n" + "=" * 60)
print("LAB 3 - BASELINE MODEL")
print("=" * 60)

print("\nProcessed data loaded successfully.")

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)


# ============================================================
# BASELINE MODEL
# ============================================================

print("\nTraining Random Forest baseline model...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(
    X_test
)


# ============================================================
# EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    predictions
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("LAB 3 BASELINE RESULTS")
print("=" * 60)

print("\nModel:")
print("RandomForestRegressor")

print("\nParameters:")
print("n_estimators:", 100)
print("random_state:", 42)

print("\nMetrics:")
print("MAE :", round(mae, 3))
print("MSE :", round(mse, 3))
print("RMSE:", round(rmse, 3))
print("R2  :", round(r2, 3))

print("\n" + "=" * 60)
print("LAB 3 COMPLETED SUCCESSFULLY")
print("=" * 60)