import os
import pickle
import json
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. Paths
# ============================================================

X_TRAIN_FILE = "artifacts/data/processed/X_train.npy"

Y_TRAIN_FILE = "artifacts/data/processed/y_train.npy"

MODEL_DIR = "artifacts/models"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "production_model.pkl"
)

METRICS_DIR = "artifacts/evaluation"

METRICS_FILE = os.path.join(
    METRICS_DIR,
    "RandomForest_metrics.json"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# ============================================================
# 2. Check processed training data
# ============================================================

if not os.path.exists(X_TRAIN_FILE):
    raise FileNotFoundError(
        f"X_train not found: {X_TRAIN_FILE}\n"
        "Run preprocess.py first."
    )

if not os.path.exists(Y_TRAIN_FILE):
    raise FileNotFoundError(
        f"y_train not found: {Y_TRAIN_FILE}\n"
        "Run preprocess.py first."
    )


# ============================================================
# 3. Load training data
# ============================================================

print("=" * 60)
print("LOADING TRAINING DATA")
print("=" * 60)

X_train = np.load(X_TRAIN_FILE)

y_train = np.load(Y_TRAIN_FILE)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


# ============================================================
# 4. Create Random Forest model
# ============================================================

print("\nCreating Random Forest model...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 5. Train model
# ============================================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ============================================================
# 6. Training predictions
# ============================================================

print("\nGenerating training predictions...")

train_predictions = model.predict(
    X_train
)


# ============================================================
# 7. Calculate training metrics
# ============================================================

mae = mean_absolute_error(
    y_train,
    train_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_train,
        train_predictions
    )
)

r2 = r2_score(
    y_train,
    train_predictions
)


# ============================================================
# 8. Save metrics
# ============================================================

metrics = {
    "model": "RandomForestRegressor",
    "n_estimators": 200,
    "random_state": 42,
    "training_mae": float(mae),
    "training_rmse": float(rmse),
    "training_r2": float(r2)
}

with open(
    METRICS_FILE,
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ============================================================
# 9. Save production model
# ============================================================

print("\nSaving production model...")

with open(
    MODEL_FILE,
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


# ============================================================
# 10. Final output
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)

print("\nModel: RandomForestRegressor")

print("\nTraining Metrics:")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

print("\nModel saved at:")
print(MODEL_FILE)

print("\nMetrics saved at:")
print(METRICS_FILE)