import json
from pathlib import Path

import joblib
import pandas as pd

from api.preprocessing import preprocess_input


MODEL_PATH = Path(
    "models/bigmart_model.pkl"
)

DATA_PATH = Path(
    "data/raw/Test.csv"
)

model = joblib.load(
    MODEL_PATH
)

df = pd.read_csv(
    DATA_PATH
)

X = preprocess_input(
    df.to_dict(
        orient="records"
    )
)

predictions = model.predict(
    X
)

report = {
    "records_monitored": int(
        len(predictions)
    ),
    "prediction_mean": float(
        predictions.mean()
    ),
    "prediction_std": float(
        predictions.std()
    ),
    "prediction_min": float(
        predictions.min()
    ),
    "prediction_max": float(
        predictions.max()
    ),
    "negative_predictions": int(
        (predictions < 0).sum()
    ),
    "zero_predictions": int(
        (predictions == 0).sum()
    ),
}

Path("artifacts").mkdir(
    parents=True,
    exist_ok=True
)

Path(
    "artifacts/prediction_monitoring_report.json"
).write_text(
    json.dumps(
        report,
        indent=2
    ),
    encoding="utf-8"
)

print(
    json.dumps(
        report,
        indent=2
    )
)