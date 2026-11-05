import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from api.preprocessing import preprocess_input


TARGET = "Item_Outlet_Sales"

BASELINE_DATA = Path(
    "data/raw/Train.csv"
)

CURRENT_DATA = Path(
    "data/monitoring/drift/concept_drift_data.csv"
)

MODEL_PATH = Path(
    "models/bigmart_model.pkl"
)


def calculate_metrics(y_true, predictions):

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = mean_squared_error(
        y_true,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_true,
        predictions
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


model = joblib.load(MODEL_PATH)

baseline_df = pd.read_csv(
    BASELINE_DATA
)

current_df = pd.read_csv(
    CURRENT_DATA
)

baseline_features = baseline_df.drop(
    columns=[TARGET]
)

current_features = current_df.drop(
    columns=[TARGET]
)

X_baseline = preprocess_input(
    baseline_features.to_dict(
        orient="records"
    )
)

X_current = preprocess_input(
    current_features.to_dict(
        orient="records"
    )
)

y_baseline = baseline_df[TARGET]
y_current = current_df[TARGET]

baseline_predictions = model.predict(
    X_baseline
)

current_predictions = model.predict(
    X_current
)

baseline_metrics = calculate_metrics(
    y_baseline,
    baseline_predictions
)

current_metrics = calculate_metrics(
    y_current,
    current_predictions
)

rmse_change = (
    (current_metrics["rmse"] -
     baseline_metrics["rmse"])
    /
    max(baseline_metrics["rmse"], 1e-9)
    * 100
)

mae_change = (
    (current_metrics["mae"] -
     baseline_metrics["mae"])
    /
    max(baseline_metrics["mae"], 1e-9)
    * 100
)

concept_drift_detected = (
    rmse_change > 10
    or mae_change > 10
)

report = {
    "baseline_metrics": baseline_metrics,
    "current_metrics": current_metrics,
    "rmse_change_percent": float(rmse_change),
    "mae_change_percent": float(mae_change),
    "threshold_percent": 10,
    "concept_drift_detected": concept_drift_detected,
}

Path("artifacts").mkdir(
    parents=True,
    exist_ok=True
)

Path(
    "artifacts/concept_drift_report.json"
).write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print(
    json.dumps(report, indent=2)
)